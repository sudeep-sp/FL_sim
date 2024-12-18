import plotly.graph_objects as go
import re
import streamlit as st
import os
import subprocess
import time
from datetime import datetime
import matplotlib.pyplot as plt
from dataset import load_datasets

# Constants
NUM_PARTITIONS = 10
LOG_DIR = "outputs"  # Base directory where logs are stored
SCRIPT_TO_RUN = "main.py"  # Path to the Federated Learning script

# Function to clean log content


def clean_log_content(raw_content):
    # Remove timestamps, log levels, and prefixes
    cleaned_content = re.sub(
        r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\]\[.*?\] - ", "", raw_content)
    return cleaned_content
# Function to parse summary and extract metrics


def parse_summary(summary_text):
    # Parse execution time
    exec_time_pattern = r"Run finished .* in ([0-9.]+)s"
    exec_time = re.search(exec_time_pattern, summary_text)
    exec_time = float(exec_time.group(1)) if exec_time else None

    # Parse distributed losses
    dist_loss_pattern = r"History \(loss, distributed\):\n((?:\s+round \d+: [0-9.]+\n)+)"
    dist_loss_data = re.findall(dist_loss_pattern, summary_text)
    distributed_losses = {}
    if dist_loss_data:
        for line in dist_loss_data[0].strip().split("\n"):
            round_num, loss = re.findall(r"round (\d+): ([0-9.]+)", line)[0]
            distributed_losses[int(round_num)] = float(loss)

    # Parse centralized losses
    cent_loss_pattern = r"History \(loss, centralized\):\n((?:\s+round \d+: [0-9.]+\n)+)"
    cent_loss_data = re.findall(cent_loss_pattern, summary_text)
    centralized_losses = {}
    if cent_loss_data:
        for line in cent_loss_data[0].strip().split("\n"):
            round_num, loss = re.findall(r"round (\d+): ([0-9.]+)", line)[0]
            centralized_losses[int(round_num)] = float(loss)

   # Parse accuracy
    accuracy_pattern = r"History \(metrics, centralized\):\n\s*{'accuracy': \[(.*?)\]}"

    accuracy_data = re.search(accuracy_pattern, summary_text)
    global_accuracy = {}
    if accuracy_data:
        accuracy_entries = accuracy_data.group(1).strip().split("),")
        for entry in accuracy_entries:
            match = re.search(r"\((\d+), ([0-9.]+)\)", entry)
            if match:
                round_num, acc = match.groups()
                global_accuracy[int(round_num)] = float(acc)
    return exec_time, distributed_losses, centralized_losses, global_accuracy


# Title and Project Description
st.title("Federated Learning Simulation")
# st.markdown("""
#     This app allows you to execute a Federated Learning simulation, visualize the dataset,
#     and analyze the generated logs in real-time.
# """)
st.write("Number of clients: ", NUM_PARTITIONS)
st.write("Number of Clients used for training: ", NUM_PARTITIONS * 0.3)
st.write("Number of Clients used for evaluation: ", NUM_PARTITIONS * 0.3)

# Display Dataset Visualization
st.header("Dataset Visualization")
trainloader, _, _ = load_datasets(
    partition_id=0, num_partitions=NUM_PARTITIONS)
batch = next(iter(trainloader))
images, labels = batch["image"], batch["label"]

# Prepare images for visualization
images = images.permute(0, 2, 3, 1).numpy() / 2 + 0.5  # Denormalize

# Display images in a grid
fig, axs = plt.subplots(4, 8, figsize=(12, 6))
for i, ax in enumerate(axs.flat):
    ax.imshow(images[i], cmap="gray")
    ax.set_title(trainloader.dataset.features["label"].int2str([labels[i]])[0])
    ax.axis("off")
fig.tight_layout()
st.pyplot(fig)


# Button to Execute Simulation
if st.button("Run Simulation"):
    st.write("Executing Federated Learning simulation...")

    # Run the simulation script
    try:
        subprocess.run(["python", SCRIPT_TO_RUN], check=True)
        st.success("Simulation executed successfully!")
    except subprocess.CalledProcessError as e:
        st.error(f"Simulation failed with error: {e}")

# Display Logs
st.header("Simulation Logs")
today = datetime.now().strftime("%Y-%m-%d")
log_dir_today = os.path.join(LOG_DIR, today)

if os.path.exists(log_dir_today):
    log_times = os.listdir(log_dir_today)
    if log_times:
        selected_time = st.selectbox("Select Log Time", log_times)
        log_file_path = os.path.join(log_dir_today, selected_time, "main.log")

        if os.path.exists(log_file_path):
            with open(log_file_path, "r") as log_file:
                raw_content = log_file.read()
                # Clean the raw log content
                cleaned_content = clean_log_content(raw_content)
            st.text_area("Log Output", cleaned_content, height=300)
        else:
            st.warning("Log file not found for the selected time.")
    else:
        st.warning("No logs available for today.")
else:
    st.warning("No logs directory available for today.")

# Input for Log Summary
st.header("Simulation Metrics")
summary_input = st.text_area(
    "Paste the summary log below (e.g., `[SUMMARY]`)", height=150)

if st.button("Visualize"):
    if summary_input.strip():
        exec_time, distributed_losses, centralized_losses, global_accuracy = parse_summary(
            summary_input
        )

        # Display Execution Time
        st.subheader("Execution Time")
        st.write(f"Run finished in {exec_time:.2f} seconds.")

        # Plot Losses
        st.subheader("Losses")
        fig1, ax1 = plt.subplots()
        ax1.plot(distributed_losses.keys(), distributed_losses.values(),
                 label="Distributed Loss", marker="o")
        ax1.plot(centralized_losses.keys(), centralized_losses.values(),
                 label="Centralized Loss", marker="o")
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss per Round")
        ax1.legend()
        st.pyplot(fig1)

        # Display Accuracy
        if global_accuracy:
            accuracy_percentage = {k: v * 100 for k,
                                   v in global_accuracy.items()}

            # Display Accuracy in Percentage
            last_round = max(accuracy_percentage.keys()
                             )  # Get the latest round
            # Accuracy for the last round
            last_accuracy = accuracy_percentage[last_round]

            st.subheader(f"Accuracy")

            # Create Circular Progress Indicator using Plotly
            fig2 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=last_accuracy,
                title={"text": "Accuracy", "font": {"size": 24}},
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": "blue"},
                    "bgcolor": "white",
                    "steps": [
                        {"range": [0, 25], "color": "red"},
                        {"range": [25, 50], "color": "orange"},
                        {"range": [50, 75], "color": "yellow"},
                        {"range": [75, 100], "color": "green"}
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": last_accuracy
                    }
                }
            ))

            # Show the gauge chart
            st.plotly_chart(fig2)

        else:
            st.warning("No accuracy data available.")

    else:
        st.error("Please paste the summary log to parse.")

st.write("🛠️ built with by Sudeep")
