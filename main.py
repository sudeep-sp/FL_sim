import torch

import hydra

from flwr.client import ClientApp
from flwr.server import ServerApp
from flwr.simulation import run_simulation

from clients import client_fn
from server import server_fn_FedAvg

from omegaconf import DictConfig


NUM_PARTITIONS = 10

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

backend_config = {"client_resources": None}

# if DEVICE.type == "mps":
#     backend_config = {"client_resources": {"num_gpus": 1}}


@hydra.main(config_path='conf', config_name='base', version_base=None)
def main(cfg: DictConfig):
    # Create ClientsApp
    client = ClientApp(client_fn=client_fn)

   # Create ServerApp
    server = ServerApp(server_fn=server_fn_FedAvg)

    # Run simulation
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_PARTITIONS,
        backend_config=backend_config,
    )


if __name__ == '__main__':
    main()
