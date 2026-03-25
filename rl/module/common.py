from typing import Tuple, Union, Any
from torch import nn
import torch

MIN_SIGMA = -20
MAX_SIGMA = 2


class Net(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int = 0,
            hidden_layers: Tuple[int, ...] = (128,),
            activation: str = 'tanh',
            output_activation: Union[None, str] = None,
    ) -> None:
        super().__init__()
        assert len(hidden_layers) >= 1
        assert activation is not None
        activation = activation.lower()
        self.input_dim = input_dim
        self.output_dim = output_dim

        net = []
        dim0 = input_dim
        for dim1 in hidden_layers:
            net.append(nn.Linear(dim0, dim1))
            net.append(get_activation(activation))
            dim0 = dim1
        if output_dim > 0:
            net.append(nn.Linear(dim0, output_dim))
            if output_activation is not None:
                output_activation = output_activation.lower()
                net.append(get_activation(output_activation))
        self.net = nn.Sequential(*net)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


def get_activation(name: str):
    if name == "elu":
        return nn.ELU(inplace=True)
    elif name == "selu":
        return nn.SELU(inplace=True)
    elif name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "crelu":
        return nn.ReLU(inplace=True)
    elif name == "lrelu":
        return nn.LeakyReLU(inplace=True)
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "softsign":
        return nn.Softsign()
    else:
        print(f"{name} is invalid activation function!")
