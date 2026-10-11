from dataclasses import dataclass

import torch

from ..geometry import Mesh


@dataclass
class MGNBatch:
    t: torch.Tensor
    dt: float
    X_ref: torch.Tensor
    u_exact: torch.Tensor
    psi_exact: torch.Tensor
    u_fixed_exact: torch.Tensor
    trac_ext: torch.Tensor


class MGNData:
    def __init__(self, dataset: dict[str, torch.Tensor], mesh: Mesh) -> None:
        self.time = dataset["time"]
        self.num_steps = len(self.time)
        self.dt = float((self.time[-1] - self.time[0]) / max(self.num_steps - 1, 1))
        self.dataset = dataset
        self.mesh = mesh

    def get_batch(self, step: int) -> MGNBatch:

        return MGNBatch(
            t=self.time[step],
            dt=self.dt,
            X_ref=self.dataset["nodes"],
            u_exact=self.dataset["u"][step],
            psi_exact=self.dataset["psi"][step],
            u_fixed_exact=self.dataset["u_fixed"][step],
            trac_ext=self.dataset["trac_ext"][step],
        )
