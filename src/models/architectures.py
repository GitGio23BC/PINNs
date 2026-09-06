import torch
from torch import nn

from src.utils import check_tensor

from .backbones import BACKBONE_REGISTRY


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

        return torch.cat([u_raw, S_raw], dim=-1)


class ParametricPINNAn(nn.Module):
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
        u_ansatz = X_coord * u_raw

        return torch.cat([u_ansatz, S_raw], dim=-1)


class ParametricPINNAn_t(nn.Module):
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


def build_model(cfg: dict[str, dict], device: torch.device) -> nn.Module:
    x_range = cfg["domain"]["x_range"]
    y_range = cfg["domain"]["y_range"]
    t_range = cfg["domain"]["t_range"]

    backbone_cls = BACKBONE_REGISTRY[cfg["model"]["architecture"]]
    backbone = backbone_cls(
        in_dim=3,
        hidden_layers=int(cfg["model"]["num_layers"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        out_dim=5,
    )

    model_type = cfg["model"].get("type", "ansatz_space").lower()

    if model_type == "standard":
        return ParametricPINN(
            backbone=backbone,
            x_range=x_range,
            y_range=y_range,
            t_range=t_range,
        ).to(device)
    elif model_type == "ansatz_space":
        return ParametricPINNAn(
            backbone=backbone,
            x_range=x_range,
            y_range=y_range,
            t_range=t_range,
        ).to(device)
    elif model_type == "ansatz_spacetime":
        return ParametricPINNAn_t(
            backbone=backbone,
            x_range=x_range,
            y_range=y_range,
            t_range=t_range,
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
