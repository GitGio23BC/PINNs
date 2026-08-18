import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_layers: int,
        hidden_dim: int,
        out_dim: int,
        activation: nn.Module = nn.Tanh(),  # noqa: B008
    ) -> None:
        super().__init__()

        self.activation = activation
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


class MMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_layers: int,
        hidden_dim: int,
        out_dim: int,
        activation: nn.Module = nn.Tanh(),  # noqa: B008
    ) -> None:
        super().__init__()

        self.activation = activation

        self.encoder_U = nn.Linear(in_dim, hidden_dim)
        self.encoder_V = nn.Linear(in_dim, hidden_dim)

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers)]
        )

        self.output_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        U = self.activation(self.encoder_U(x))
        V = self.activation(self.encoder_V(x))

        H = U
        for layer in self.hidden_layers:
            Z = self.activation(layer(H))
            H = (1 - Z) * U + Z * V

        out = self.output_layer(H)
        return out

BACKBONE_REGISTRY = {
    "MLP": MLP,
    "MMLP": MMLP,
}
