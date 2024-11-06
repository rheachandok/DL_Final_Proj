from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
        input_dim=None,
        arch_subclass: str = "a",
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        if arch == "conv":
            self.prober = build_conv(
                PROBER_CONV_LAYERS_CONFIG[arch_subclass], input_dim=input_dim
            )
        else:
            arch_list = list(map(int, arch.split("-"))) if arch != "" else []
            f = [embedding] + arch_list + [self.output_dim]
            layers = []
            for i in range(len(f) - 2):
                layers.append(torch.nn.Linear(f[i], f[i + 1]))
                # layers.append(torch.nn.BatchNorm1d(f[i + 1]))
                layers.append(torch.nn.ReLU(True))
            layers.append(torch.nn.Linear(f[-2], f[-1]))
            self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        if self.arch == "conv":
            output = self.prober(e)
        else:
            e = flatten_conv_output(e)
            output = self.prober(e)

        return output.view(*output.shape[:-1], *self.output_shape)
