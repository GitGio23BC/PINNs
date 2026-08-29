from pathlib import Path

import torch


def train_points(
    cfg: dict, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_pts = int(cfg["domain"]["n_collocation"])
    x_min, x_max = cfg["domain"]["x_range"]
    y_min, y_max = cfg["domain"]["y_range"]
    t_min, t_max = cfg["domain"]["t_range"]
    F_min, F_max = cfg["domain"]["F_range"]

    x_coords = torch.rand((n_pts, 1), device=device) * (x_max - x_min) + x_min
    y_coords = torch.rand((n_pts, 1), device=device) * (y_max - y_min) + y_min
    x = torch.cat([x_coords, y_coords], dim=-1)

    t = torch.rand((n_pts, 1), device=device) * (t_max - t_min) + t_min

    F = torch.rand((n_pts, 1), device=device) * (F_max - F_min) + F_min

    return x, t, F


def initial_points(
    cfg: dict, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_pts = int(cfg["domain"]["n_ic"])
    x_min, x_max = cfg["domain"]["x_range"]
    y_min, y_max = cfg["domain"]["y_range"]
    F_min, F_max = cfg["domain"]["F_range"]

    x_coords = torch.rand((n_pts, 1), device=device) * (x_max - x_min) + x_min
    y_coords = torch.rand((n_pts, 1), device=device) * (y_max - y_min) + y_min
    x_ic = torch.cat([x_coords, y_coords], dim=-1)

    t_ic = torch.zeros((n_pts, 1), device=device)

    F_ic = torch.rand((n_pts, 1), device=device) * (F_max - F_min) + F_min

    return x_ic, t_ic, F_ic


def boundary_points(
    cfg: dict, device: torch.device
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    n_bc = int(cfg["domain"]["n_bc"]) // 4
    x_min, x_max = cfg["domain"]["x_range"]
    y_min, y_max = cfg["domain"]["y_range"]
    t_min, t_max = cfg["domain"]["t_range"]
    F_min, F_max = cfg["domain"]["F_range"]

    y_rand = torch.rand((n_bc, 1), device=device) * (y_max - y_min) + y_min
    x_rand = torch.rand((n_bc, 1), device=device) * (x_max - x_min) + x_min

    x_left = torch.cat([torch.full((n_bc, 1), x_min, device=device), y_rand], dim=-1)
    t_left = torch.rand((n_bc, 1), device=device) * (t_max - t_min) + t_min
    F_left = torch.rand((n_bc, 1), device=device) * (F_max - F_min) + F_min
    left = (x_left, t_left, F_left)

    x_right = torch.cat([torch.full((n_bc, 1), x_max, device=device), y_rand], dim=-1)
    t_right = torch.rand((n_bc, 1), device=device) * (t_max - t_min) + t_min
    F_right = torch.rand((n_bc, 1), device=device) * (F_max - F_min) + F_min
    right = (x_right, t_right, F_right)

    y_bottom = torch.cat([x_rand, torch.full((n_bc, 1), y_min, device=device)], dim=-1)
    t_bottom = torch.rand((n_bc, 1), device=device) * (t_max - t_min) + t_min
    F_bottom = torch.rand((n_bc, 1), device=device) * (F_max - F_min) + F_min
    bottom = (y_bottom, t_bottom, F_bottom)

    y_top = torch.cat([x_rand, torch.full((n_bc, 1), y_max, device=device)], dim=-1)
    t_top = torch.rand((n_bc, 1), device=device) * (t_max - t_min) + t_min
    F_top = torch.rand((n_bc, 1), device=device) * (F_max - F_min) + F_min
    top = (y_top, t_top, F_top)

    return left, right, bottom, top


def generate_test_data(cfg: dict, save_path: Path) -> dict[str, torch.Tensor]:
    x_min, x_max = cfg["domain"]["x_range"]
    y_min, y_max = cfg["domain"]["y_range"]
    t_min, t_max = cfg["domain"]["t_range"]
    F_min, F_max = cfg["domain"]["F_range"]

    nx = int(cfg["test"]["nx"])
    ny = int(cfg["test"]["ny"])
    nt = int(cfg["test"]["nt"])
    nF = int(cfg["test"]["nF"])

    x_grid = torch.linspace(x_min, x_max, nx)
    y_grid = torch.linspace(y_min, y_max, ny)
    t_grid = torch.linspace(t_min, t_max, nt)
    F_grid = torch.linspace(F_min, F_max, nF)

    mesh_x, mesh_y, mesh_t, mesh_F = torch.meshgrid(
        x_grid, y_grid, t_grid, F_grid, indexing="ij"
    )

    x_val = torch.stack([mesh_x.flatten(), mesh_y.flatten()], dim=-1)
    t_val = mesh_t.flatten().unsqueeze(-1)
    F_val = mesh_F.flatten().unsqueeze(-1)

    val_data = {
        "x": x_val,
        "t": t_val,
        "F": F_val,
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(val_data, save_path)
    return val_data


def load_test_data(
    cfg: dict, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    val_file = Path(cfg["data"]["data_dir"]) / "validation_grid_2D.pt"

    if not val_file.exists():
        data = generate_test_data(cfg, val_file)
    else:
        data = torch.load(val_file, map_location=device, weights_only=True)

    x_val = data["x"].to(device)
    t_val = data["t"].to(device)
    F_val = data["F"].to(device)

    return x_val, t_val, F_val
