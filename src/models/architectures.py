import torch
from torch import nn

from src.utils import check_tensor


class ParametricPINN(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        x_ref: float = 1.0,
        F_ref: float = 10.0,
        A_ref: float = 2.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.register_buffer("x_ref", torch.tensor(float(x_ref)))
        self.register_buffer("F_ref", torch.tensor(float(F_ref)))
        self.register_buffer("A_ref", torch.tensor(float(A_ref)))

    def forward(
        self, x: torch.Tensor, F: torch.Tensor, A: torch.Tensor
    ) -> torch.Tensor:
        x_norm = x / self.x_ref  # type: ignore
        F_norm = F / self.F_ref  # type: ignore
        A_norm = A / self.A_ref  # type: ignore

        N = x_norm.shape[0]
        device = x_norm.device

        F_norm = check_tensor(F_norm, N, device)
        A_norm = check_tensor(A_norm, N, device)

        in_features = torch.cat([x_norm, F_norm, A_norm], dim=-1)
        return self.backbone(in_features)
