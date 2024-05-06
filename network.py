import pickle
import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list[int] = [256, 256],
        activation_function: nn.Module = nn.ReLU(),
        output_function: nn.Module = nn.Identity(),
        device: torch.device = "cuda",
    ) -> None:
        super(Network, self).__init__()
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        layers: list[nn.Module] = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(
                activation_function if i < len(layer_sizes) - 2 else output_function
            )
        self.network = nn.Sequential(*layers)

        self.device = device
        self.to(self.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

    def save(self, save_path: str) -> None:
        pickle.dump(self.network, open(save_path, "wb"))
