import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_layers: int,
        hidden_dim: int,
        out_dim: int,
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()

        self.activation = activation()
        self.input_layer = nn.Linear(in_dim, hidden_dim)

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers)]
        )

        self.output_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.input_layer(x))

        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        x = self.output_layer(x)
        return x


BACKBONE_REGISTRY = {
    "MLP": MLP,
}