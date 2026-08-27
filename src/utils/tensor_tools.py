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

    n = x.shape[-1]
    indices = [(i, i) for i in range(n)]

    for j in range(n - 1, 0, -1):
        for i in range(j - 1, -1, -1):
            indices.append((i, j))

    voigt = torch.stack(
        [x[..., i, j] for i, j in indices],
        dim=-1,
    )

    if is_shear:
        diagonal = voigt[..., :n]
        shear = 2.0 * voigt[..., n:]
        voigt = torch.cat((diagonal, shear), dim=-1)

    return voigt


def div(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
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

def d_dt(E: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    d = E.shape[-1]
    rows = []
    for i in range(d):
        cols = []
        for j in range(d):
            dE_ij_dt = torch.autograd.grad(
                E[:, i, j],
                t,
                grad_outputs=torch.ones_like(E[:, i, j]),
                create_graph=True,
                retain_graph=True,
            )[0]  
            cols.append(dE_ij_dt)
        rows.append(torch.cat(cols, dim=-1)) 
    return torch.stack(rows, dim=1) 