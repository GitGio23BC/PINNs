from dataclasses import dataclass

import torch


@dataclass
class MGNBatch:
    t: torch.Tensor
    dt: float
    X_ref: torch.Tensor
    u: torch.Tensor
    E: torch.Tensor
    S: torch.Tensor
    trac: torch.Tensor


class MGNData:
    def __init__(self, cfg: dict, dataset: dict[str, torch.Tensor]) -> None:
        self.time = dataset["time"]
        self.num_steps = len(self.time)
        self.dt = float((self.time[-1] - self.time[0]) / max(self.num_steps - 1, 1))
        self.dataset = dataset

    def get_batch(self, step: int) -> MGNBatch:
        return MGNBatch(
            t=self.time[step],
            dt=self.dt,
            X_ref=self.dataset["nodes"],
            u=self.dataset["u"][step],
            E=self.dataset["E"][step],
            S=self.dataset["S"][step],
            trac=self.dataset["trac_ext"][step],
        )
