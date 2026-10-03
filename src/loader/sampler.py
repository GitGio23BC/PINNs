from dataclasses import dataclass

import torch


@dataclass
class MGNBatch:
    t: torch.Tensor
    dt: float
    X_ref: torch.Tensor
    u: torch.Tensor
    F_ext: torch.Tensor


class MGNData:
    def __init__(self, dataset: dict[str, torch.Tensor]) -> None:
        self.time = dataset["time"]
        self.num_steps = len(self.time)
        self.dt = float((self.time[-1] - self.time[0]) / max(self.num_steps - 1, 1))
        self.dataset = dataset

    def load_physics(
        self,
        mus: torch.Tensor,
        alphas: torch.Tensor,
        beta: float,
        lam: float,
    ):
        pass

    def get_batch(self, step: int) -> MGNBatch:
        return MGNBatch(
            t=self.time[step],
            dt=self.dt,
            X_ref=self.dataset["nodes"],
            u=self.dataset["u"][step],
            F_ext=self.dataset["F_ext"][step],
        )
