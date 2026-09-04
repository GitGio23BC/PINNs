from dataclasses import dataclass

import torch

from ..geometry import Mesh


@dataclass
class PINNBatch:
    t_res: torch.Tensor
    X_res: torch.Tensor

    t_ic: torch.Tensor
    X_ic: torch.Tensor
    u_ic_target: torch.Tensor
    S_ic_target: torch.Tensor

    t_base: torch.Tensor
    X_base: torch.Tensor

    t_neu: torch.Tensor
    X_neu: torch.Tensor
    normals_neu: torch.Tensor
    trac_target: torch.Tensor


class PINNSampler:
    def __init__(
        self,
        dataset: dict[str, torch.Tensor],
        cfg: dict,
        mesh: Mesh,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = device
        self.cfg = cfg

        self.time = dataset["time"].to(device)
        self.nodes = dataset["nodes"].to(device)
        self.u = dataset["u"].to(device)
        self.S = dataset["S"].to(device)
        self.trac_ext = dataset["trac_ext"].to(device)

        self.left_nodes = mesh.left_nodes.to(device)
        self.right_nodes = mesh.right_nodes.to(device)
        self.bottom_nodes = mesh.bottom_nodes.to(device)
        self.top_nodes = mesh.top_nodes.to(device)

        self.sigma_val = float(cfg["physics"].get("sigma_max", 1.0))

    def sample_window(
        self,
        step_start: int,
        step_end: int,
        n_continuous_points: int = 0,
    ) -> PINNBatch:

        t_window = self.time[step_start:step_end]
        W = t_window.shape[0]
        N_nodes = self.nodes.shape[0]

        # Residual Points
        X_res_mesh = self.nodes.repeat(W, 1)
        t_res_mesh = t_window.repeat_interleave(N_nodes).unsqueeze(-1)

        if n_continuous_points > 0:
            x_min, x_max = self.cfg["domain"]["x_range"]
            y_min, y_max = self.cfg["domain"]["y_range"]
            t_min, t_max = t_window[0].item(), t_window[-1].item()

            X_rand = torch.empty(n_continuous_points, 2, device=self.device)
            X_rand[:, 0].uniform_(x_min, x_max)
            X_rand[:, 1].uniform_(y_min, y_max)

            t_rand = torch.empty(
                n_continuous_points,
                1,
                device=self.device,
            ).uniform_(t_min, t_max)

            X_res = torch.cat([X_res_mesh, X_rand], dim=0)
            t_res = torch.cat([t_res_mesh, t_rand], dim=0)
        else:
            X_res = X_res_mesh
            t_res = t_res_mesh

        X_res = X_res.clone().detach().requires_grad_(True)
        t_res = t_res.clone().detach().requires_grad_(True)

        # Inital Condition Points
        X_ic = self.nodes.clone().detach()
        t_ic = self.time[step_start].expand(N_nodes, 1).clone().detach()
        u_ic_target = self.u[step_start].clone().detach()
        S_ic_target = self.S[step_start].clone().detach()

        # Dirichlet Points
        n_left = self.left_nodes.shape[0]
        X_base = self.nodes[self.left_nodes].repeat(W, 1).clone().detach()
        t_base = t_window.repeat_interleave(n_left).unsqueeze(-1).clone().detach()

        # Newmann Points
        neu_groups = [
            (self.right_nodes, torch.tensor([1.0, 0.0], device=self.device)),
            (self.top_nodes, torch.tensor([0.0, 1.0], device=self.device)),
            (self.bottom_nodes, torch.tensor([0.0, -1.0], device=self.device)),
        ]

        X_neu_list, t_neu_list, n_neu_list, trac_neu_list = [], [], [], []
        for idx_group, normal_vec in neu_groups:
            k = idx_group.shape[0]
            X_neu_list.append(self.nodes[idx_group].repeat(W, 1))
            t_neu_list.append(t_window.repeat_interleave(k).unsqueeze(-1))
            n_neu_list.append(normal_vec.expand(k * W, 2))

            trac_group = self.trac_ext[step_start:step_end, idx_group, :].reshape(-1, 2)
            trac_neu_list.append(trac_group)

        X_neu = torch.cat(X_neu_list, dim=0).clone().detach().requires_grad_(True)
        t_neu = torch.cat(t_neu_list, dim=0).clone().detach()
        normals_neu = torch.cat(n_neu_list, dim=0)
        trac_target = torch.cat(trac_neu_list, dim=0)

        return PINNBatch(
            t_res=t_res,
            X_res=X_res,
            t_ic=t_ic,
            X_ic=X_ic,
            u_ic_target=u_ic_target,
            S_ic_target=S_ic_target,
            t_base=t_base,
            X_base=X_base,
            t_neu=t_neu,
            X_neu=X_neu,
            normals_neu=normals_neu,
            trac_target=trac_target,
        )
