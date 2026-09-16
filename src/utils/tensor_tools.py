import math

import torch


def check_tensor(
    x: float | torch.Tensor, N: int, device: torch.device | str
) -> torch.Tensor:
    if isinstance(x, (float, int)):
        return torch.full((N, 1), float(x), device=device, dtype=torch.float32)

    if isinstance(x, torch.Tensor):
        if x.dim() == 0 or (x.dim() == 1 and x.numel() == 1):
            return x.expand(N, 1).to(device)
        if x.dim() == 1 and x.shape[0] == N:
            return x.unsqueeze(-1).to(device)
        return x.to(device)

    raise TypeError(f"Unsupported input type for tensor conversion: {type(x)}")


def voigt_tensor(
    x: torch.Tensor,
    is_shear: bool = False,
) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError("Input must have at least two dimensions")

    if x.shape[-1] != x.shape[-2]:
        raise ValueError("Square tensor needed")

    d = x.shape[-1]

    diag_indices = [(i, i) for i in range(d)]
    row_idx, col_idx = torch.triu_indices(d, d, offset=1)
    off_diag_indices = list(zip(row_idx.tolist(), col_idx.tolist()))

    all_indices = diag_indices + off_diag_indices

    voigt = torch.stack(
        [x[..., i, j] for i, j in all_indices],
        dim=-1,
    )

    if is_shear:
        diagonal = voigt[..., :d]
        shear = 2.0 * voigt[..., d:]
        voigt = torch.cat((diagonal, shear), dim=-1)

    return voigt


def voigt_to_tensor(
    x_voigt: torch.Tensor,
    is_shear: bool = False,
) -> torch.Tensor:
    if x_voigt.ndim < 2:
        raise ValueError("Input must have at least two dimensions")

    v = x_voigt.squeeze(-1) if x_voigt.ndim == 3 else x_voigt
    N, n = v.shape

    d = int((-1 + math.sqrt(1 + 8 * n)) / 2)

    x = torch.zeros(
        N,
        d,
        d,
        dtype=v.dtype,
        device=v.device,
    )

    diagonal = torch.arange(d, device=v.device)
    x[:, diagonal, diagonal] = v[:, :d]

    row_idx, col_idx = torch.triu_indices(d, d, offset=1, device=v.device)
    shear = v[:, d:]

    factor = 2.0 if is_shear else 1.0
    x[:, row_idx, col_idx] = shear / factor
    x[:, col_idx, row_idx] = shear / factor

    return x


def div(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1 and None: # TO DO #
        nx, ny = x
        y_grid = y.view(int(ny.item()) + 1, int(nx.item() + 1), 2)
        dx = dy = 1

        div_y_X = (y_grid[1:-1, 2:, :, 0] - y_grid[1:-1, :-2, :, 0]) / (2.0 * dx)
        div_y_Y = (y_grid[2:, 1:-1, :, 1] - y_grid[:-2, 1:-1, :, 1]) / (2.0 * dy)

        div_y = div_y_X + div_y_Y  
        return div_y.reshape(-1, 2)

    d = x.shape[-1]
    div_rows = []

    for i in range(d):
        row_div = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        for j in range(d):
            dy_ij = torch.autograd.grad(
                y[:, i, j],
                x,
                grad_outputs=torch.ones_like(y[:, i, j]),
                create_graph=True,
                retain_graph=True,
            )[0]
            row_div = row_div + dy_ij[:, j : j + 1]
        div_rows.append(row_div)

    return torch.cat(div_rows, dim=-1)


def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1 and None: # TO DO #
        nx, ny = x
        d = y.shape[-1]
        y_grid = y.view(int(ny.item()) + 1, int(nx.item() + 1), 2)
        dx = dy = 1

        dy_dX = (y_grid[1:-1, 2:, :] - y_grid[1:-1, :-2, :]) / (2.0 * dx)
        dy_dY = (y_grid[2:, 1:-1, :] - y_grid[:-2, 1:-1, :]) / (2.0 * dy)

        grad_y = torch.stack([dy_dX, dy_dY], dim=-1)  
        return grad_y.reshape(-1, d, 2)

    d = x.shape[-1]
    rows = []

    for i in range(d):
        dy_i = torch.autograd.grad(
            y[:, i],
            x,
            grad_outputs=torch.ones_like(y[:, i]),
            create_graph=True,
            retain_graph=True,
        )[0]
        rows.append(dy_i)

    return torch.stack(rows, dim=1)


def d_dt(E_voigt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    E_dot_components = []
    for i in range(3):
        dE_i = torch.autograd.grad(
            E_voigt[:, i],
            t,
            grad_outputs=torch.ones_like(E_voigt[:, i]),
            create_graph=True,
            retain_graph=True,
        )[0]
        E_dot_components.append(dE_i)

    E_dot_voigt = torch.cat(E_dot_components, dim=-1)
    return E_dot_voigt
