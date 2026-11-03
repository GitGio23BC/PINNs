from dataclasses import dataclass

import torch

from ..geometry import Mesh


@dataclass
class LossScales:
    u_scale: torch.Tensor | float
    trac_scale: torch.Tensor | float
    force_scale: torch.Tensor | float
    psi_scale: torch.Tensor | float


@dataclass
class BodyState:
    b: torch.Tensor
    rho: torch.Tensor
    u: torch.Tensor
    u_t: torch.Tensor
    u_tt: torch.Tensor
    P: torch.Tensor
    psi: torch.Tensor


@dataclass
class MGNBatch:
    t: torch.Tensor
    dt: float
    X_ref: torch.Tensor
    u_exact: torch.Tensor
    psi_exact: torch.Tensor
    u_fixed_exact: torch.Tensor
    trac_ext: torch.Tensor
    F_ext: torch.Tensor


class MGNData:
    def __init__(
        self,
        dataset: dict[str, torch.Tensor],
        mesh: Mesh,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.dataset = {
            key: value.to(device=device, dtype=dtype)
            if isinstance(value, torch.Tensor) and value.is_floating_point()
            else value.to(device=device)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in dataset.items()
        }

        self.time = self.dataset["time"]
        self.num_steps = len(self.time)
        self.dt = float((self.time[-1] - self.time[0]) / max(self.num_steps - 1, 1))
        self.mesh = mesh
        self.device = device
        self.dtype = dtype

        trac_ext = self.dataset["trac_ext"]

        N = trac_ext.shape[0]

        F_ext = torch.zeros((N, self.mesh.n_nodes, 2), dtype=dtype, device=device)
        F_ext[:, self.mesh.right_nodes, 0] = trac_ext[:, :, 0] * self.mesh.right_lengths
        self.dataset["F_ext"] = F_ext

    def get_batch(self, step: int) -> MGNBatch:

        return MGNBatch(
            t=self.time[step].detach(),
            dt=self.dt,
            X_ref=self.dataset["nodes"].detach(),
            u_exact=self.dataset["u"][step].detach(),
            psi_exact=self.dataset["psi"][step].detach(),
            u_fixed_exact=self.dataset["u_fixed"][step].detach(),
            trac_ext=self.dataset["trac_ext"][step].detach(),
            F_ext=self.dataset["F_ext"][step].detach(),
        )
