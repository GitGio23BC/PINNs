import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_layers: int,
        hidden_dim: int,
        out_dim: int,
        use_layer_norm: bool = True,
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()

        self.activation = activation()
        self.input_layer = nn.Linear(in_dim, hidden_dim)

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers)]
        )

        layers = [nn.Linear(hidden_dim, out_dim)]

        if use_layer_norm:
            layers.append(nn.LayerNorm(out_dim)) # type: ignore

        self.output_layer = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.input_layer(x))

        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        x = self.output_layer(x)
        return x


ACTIVATION_REGISTRY: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}

BACKBONE_REGISTRY = {
    "MLP": MLP,
}
