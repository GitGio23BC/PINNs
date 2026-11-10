import math

import torch


def check_tensor(
    x: float | torch.Tensor, N: int, device: torch.device | str
) -> torch.Tensor:
    if isinstance(x, (float, int)):
        return torch.full((N, 1), float(x), device=device, dtype=torch.float32)

    if isinstance(x, torch.Tensor):
        if x.dim() == 0 or (x.dim() == 1 and x.numel() == 1):
            return x.expand(N, 1).to(device=device, dtype=torch.float32)
        if x.dim() == 1 and x.shape[0] == N:
            return x.unsqueeze(-1).to(device=device, dtype=torch.float32)
        return x.to(device=device, dtype=torch.float32)

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


def tetrahedral_shape_gradients(
    nodes: torch.Tensor, elements: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    p0 = nodes[elements[:, 0]]
    p1 = nodes[elements[:, 1]]
    p2 = nodes[elements[:, 2]]
    p3 = nodes[elements[:, 3]]

    e1 = p1 - p0
    e2 = p2 - p0
    e3 = p3 - p0

    J = torch.stack((e1, e2, e3), dim=-1)  
    invJ = torch.linalg.inv(J)

    grad123 = invJ 

    grad0 = -grad123.sum(dim=1, keepdim=True)
    dN_dX = torch.cat((grad0, grad123), dim=1)  

    volumes = torch.abs(torch.linalg.det(torch.stack([e1, e2, e3], dim=1))) / 6.0


    return dN_dX, volumes


def stencil_grad(
    y: torch.Tensor, elements: torch.Tensor, dN_dX: torch.Tensor
) -> torch.Tensor:

    # Get each element value
    y_elem = y[elements]

    # grad_y = u_elem^T @ dN_dX for each node a
    grad_u = torch.einsum("eai, eaj -> eij", y_elem, dN_dX)
    return grad_u
