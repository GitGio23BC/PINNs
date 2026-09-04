import torch
from torch import nn

from src.utils import check_tensor


class ParametricPINN(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        x_range: tuple[float, float] = (0.0, 1.0),
        y_range: tuple[float, float] = (0.0, 1.0),
        t_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        super().__init__()
        self.backbone = backbone

        self.register_buffer("X_min", torch.tensor([x_range[0], y_range[0]]))
        self.register_buffer("X_max", torch.tensor([x_range[1], y_range[1]]))
        self.register_buffer("t_min", torch.tensor(t_range[0]))
        self.register_buffer("t_max", torch.tensor(t_range[1]))

    def forward(
        self,
        X: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:

        N = X.shape[0]
        t_col = check_tensor(t, N, X.device)

        t_norm = 2.0 * (t_col - self.t_min) / (self.t_max - self.t_min) - 1.0  # type: ignore
        X_norm = 2.0 * (X - self.X_min) / (self.X_max - self.X_min) - 1.0  # type: ignore

        in_features = torch.cat([t_norm, X_norm], dim=-1)
        raw = self.backbone(in_features)
        u_raw = raw[:, :2]
        S_raw = raw[:, 2:]

        X_coord = X[:, 0:1]
        u_ansatz = t_col * X_coord * u_raw
        S_ansatz = t_col * S_raw

        return torch.cat([u_ansatz, S_ansatz], dim=-1)
