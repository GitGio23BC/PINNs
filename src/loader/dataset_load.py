from pathlib import Path

import torch


def train_points(cfg: dict, device: torch.device) -> torch.Tensor:
    n_pts = int(cfg["domain"]["n_collocation"])
    x_min, x_max = cfg["domain"]["x_range"]
    y_min, y_max = cfg["domain"]["y_range"]

    x_coords = torch.rand((n_pts, 1), device=device) * (x_max - x_min) + x_min
    y_coords = torch.rand((n_pts, 1), device=device) * (y_max - y_min) + y_min
    x = torch.cat([x_coords, y_coords], dim=-1)

    return x


def initial_points(cfg: dict, device: torch.device) -> torch.Tensor:
    n_ic = int(cfg["domain"]["n_ic"])
    x_min, x_max = cfg["domain"]["x_range"]
    y_min, y_max = cfg["domain"]["y_range"]

    x_coords = torch.rand((n_ic, 1), device=device) * (x_max - x_min) + x_min
    y_coords = torch.rand((n_ic, 1), device=device) * (y_max - y_min) + y_min
    x_ic = torch.cat([x_coords, y_coords], dim=-1)

    return x_ic


def boundary_points(
    cfg: dict, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_bc = int(cfg["domain"]["n_bc"]) // 4
    x_min, x_max = cfg["domain"]["x_range"]
    y_min, y_max = cfg["domain"]["y_range"]

    y_rand = torch.rand((n_bc, 1), device=device) * (y_max - y_min) + y_min
    left = torch.cat([torch.full((n_bc, 1), x_min, device=device), y_rand], dim=-1)
    right = torch.cat([torch.full((n_bc, 1), x_max, device=device), y_rand], dim=-1)

    x_rand = torch.rand((n_bc, 1), device=device) * (x_max - x_min) + x_min
    bottom = torch.cat([x_rand, torch.full((n_bc, 1), y_min, device=device)], dim=-1)
    top = torch.cat([x_rand, torch.full((n_bc, 1), y_max, device=device)], dim=-1)

    return left, right, bottom, top


def generate_test_data(cfg: dict, save_path: Path) -> dict[str, torch.Tensor]:
    x_min, x_max = cfg["domain"]["x_range"]
    y_min, y_max = cfg["domain"]["y_range"]

    nx, ny = 20, 20
    x_grid = torch.linspace(x_min, x_max, nx)
    y_grid = torch.linspace(y_min, y_max, ny)

    mesh_x, mesh_y = torch.meshgrid(x_grid, y_grid, indexing="ij")

    x_val = torch.stack([mesh_x.flatten(), mesh_y.flatten()], dim=-1)

    val_data = {"x": x_val}

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(val_data, save_path)
    return val_data


def load_test_data(cfg: dict, device: torch.device) -> torch.Tensor:
    val_file = Path(cfg["data"]["data_dir"]) / "validation_grid_2D.pt"

    if not val_file.exists():
        data = generate_test_data(cfg, val_file)
    else:
        data = torch.load(val_file, map_location=device)

    x_val = data["x"].to(device)
    return x_val
