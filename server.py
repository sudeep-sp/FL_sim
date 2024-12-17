import torch

from flwr.common import ndarrays_to_parameters, Context
from flwr.server import ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, FedAdagrad
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar, Context

from typing import Dict, Optional, Tuple


from model import Net, get_parameters, set_parameters, test
from dataset import load_datasets

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

NUM_PARTITIONS = 10
fraction_fit = 0.3
fraction_evaluate = 0.3
min_fit_clients = 3
min_evaluate_clients = 3

params = get_parameters(Net())

# The `evaluate` function will be called by Flower after every round


def evaluate(
    server_round: int,
    parameters: NDArrays,
    config: Dict[str, Scalar],
) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    net = Net().to(DEVICE)
    _, _, testloader = load_datasets(0, NUM_PARTITIONS)
    set_parameters(net, parameters)  # Update model with the latest parameters
    loss, accuracy = test(net, testloader)
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
        "local_epochs": 1 if server_round < 2 else 2,
    }
    return config


def server_fn_FedAvg(context: Context) -> ServerAppComponents:
    # Create FedAvg strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=NUM_PARTITIONS,
        initial_parameters=ndarrays_to_parameters(
            params
        ),  # Pass initial model parameters

        evaluate_fn=evaluate,
        on_fit_config_fn=fit_config,  # Pass the fit_config function
    )

    # Configure the server for 3 rounds of training
    config = ServerConfig(num_rounds=3)
    return ServerAppComponents(strategy=strategy, config=config)


def server_fn_FedAdagrad(context: Context) -> ServerAppComponents:
    # Create FedAdagrad strategy
    strategy = FedAdagrad(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=NUM_PARTITIONS,
        initial_parameters=ndarrays_to_parameters(params),
        evaluate_fn=evaluate,
        on_fit_config_fn=fit_config,  # Pass the fit_config function
    )
    # Configure the server for 3 rounds of training
    config = ServerConfig(num_rounds=3)
    return ServerAppComponents(strategy=strategy, config=config)
