import logging
import socket
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import numpy as np
import torch

from src.geometry import Graph, create_mesh_3D
from src.loader import MGNData, generate_inertial_dataset
from src.models import MeshGraphNet
from src.utils import init_logging, load_config

UNITY_IP = "192.168.1.117"
UNITY_UDP_PORT = 5005


def stream_predictions(model_path: Path | None = None) -> None:
    init_logging()
    logger = logging.getLogger(__name__)

    cfg = load_config(config_path="config.yaml")
    device = torch.device(cfg["training"]["device"])
    dtype = torch.float32

    model_cfg = cfg["model"]
    if model_path is None:
        # Nascondi la finestra principale di tkinter
        root = tk.Tk()
        root.withdraw()
        
        # Apri il popup per selezionare il file
        print("In attesa della selezione del modello...")
        file_selezionato = filedialog.askopenfilename(
            title="Seleziona il modello addestrato (.pt)",
            filetypes=[("Modelli PyTorch", "*.pt"), ("Tutti i file", "*.*")]
        )
        
        if not file_selezionato:
            raise FileNotFoundError("Nessun file selezionato. Operazione annullata.")
        
        model_file = Path(file_selezionato)
    else:
        model_file = Path(model_path)
        
    x_min, x_max = cfg["domain"]["x_range"]
    y_min, y_max = cfg["domain"]["y_range"]
    z_min, z_max = cfg["domain"]["z_range"]
    nx, ny, nz = (
        int(cfg["domain"]["nx"]),
        int(cfg["domain"]["ny"]),
        int(cfg["domain"]["nz"]),
    )
    mesh = create_mesh_3D(
        width=float(x_max - x_min),
        height=float(y_max - y_min),
        depth=float(z_max - z_min),
        nx=nx,
        ny=ny,
        nz=nz,
        device=device,
        dtype=dtype,
    )

    output_dim = int(model_cfg["output_dim"])
    edge_in_dim = int(model_cfg["edge_in_dim"])
    spatial_dim = (edge_in_dim - 2) // 2
    node_type_classes = int(model_cfg["node_in_dim"]) - 2 * output_dim - 3
    if output_dim not in (2, 3) or spatial_dim not in (2, 3):
        raise ValueError(
            "The model must use output_dim 2 or 3 and edge_in_dim 6 or 8."
        )
    if node_type_classes not in (3, 6):
        raise ValueError(
            "The configured node_in_dim is incompatible with displacement, "
            "velocity, acceleration, and traction features."
        )

    raw_node_type = torch.zeros(mesh.n_nodes, dtype=torch.long, device=device)
    if node_type_classes == 3:
        raw_node_type[mesh.get_boundary_nodes("left")] = 1
        raw_node_type[mesh.get_boundary_nodes("right")] = 2
    else:
        for node_type_idx, boundary_nodes in enumerate(mesh.boundary_nodes.values()):
            raw_node_type[boundary_nodes] = node_type_idx
    node_type = torch.nn.functional.one_hot(
        raw_node_type, num_classes=node_type_classes
    ).to(dtype=dtype)

    data_dir = Path(cfg["data"]["data_dir"])
    dataset_path = data_dir / str(cfg["data"]["dataset_name"])
    if dataset_path.exists():
        dataset = torch.load(
            dataset_path,
            map_location=device,
            weights_only=False,
        )
    else:
        dataset = generate_inertial_dataset(cfg=cfg, device=device, dtype=dtype)

    data_loader = MGNData(
        dataset=dataset,
        mesh=mesh,
        device=device,
        dtype=dtype,
    )
    if data_loader.num_steps == 0:
        raise ValueError("The dataset contains no time steps to stream.")

    checkpoint = torch.load(
        model_file,
        map_location=device,
        weights_only=False,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    mgn = MeshGraphNet(
        node_in_dim=int(model_cfg["node_in_dim"]),
        edge_in_dim=int(model_cfg["edge_in_dim"]),
        latent_dim=int(model_cfg["latent_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        output_dim=int(model_cfg["output_dim"]),
    ).to(device=device, dtype=dtype)
    mgn.load_state_dict(state_dict)
    mgn.eval()
    logger.info("Loaded model checkpoint: %s", model_file)

    trac_scale = float(cfg["physics"]["sigma_max"])
    dt = data_loader.dt
    if dt <= 0:
        raise ValueError(f"Dataset time step must be positive, got {dt}.")

    u_prev = torch.zeros((mesh.n_nodes, output_dim), dtype=dtype, device=device)
    u_prev2 = torch.zeros_like(u_prev)
    velocity_prev = torch.zeros_like(u_prev)
    acceleration_prev = torch.zeros_like(u_prev)
    world_pos = mesh.nodes.clone()
    reference_pos = mesh.nodes[:, :spatial_dim]
    senders = torch.cat((mesh.edges[:, 0], mesh.edges[:, 1]))
    receivers = torch.cat((mesh.edges[:, 1], mesh.edges[:, 0]))
    reference_rel = reference_pos[senders] - reference_pos[receivers]

    logger.info(
        "Streaming XYZ coordinates to %s:%d",
        UNITY_IP,
        UNITY_UDP_PORT,
    )
    frame_index = 0
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_socket:
        while True:
            batch = data_loader.get_batch(frame_index % data_loader.num_steps)
            trac_feat = torch.zeros((mesh.n_nodes, 3), device=device, dtype=dtype)
            trac_feat[mesh.get_boundary_nodes("right")] = batch.trac_ext / trac_scale

            current_pos = world_pos[:, :spatial_dim]
            current_rel = current_pos[senders] - current_pos[receivers]
            edge_features = torch.cat(
                (
                    reference_rel,
                    torch.linalg.vector_norm(
                        reference_rel, dim=-1, keepdim=True
                    ),
                    current_rel,
                    torch.linalg.vector_norm(current_rel, dim=-1, keepdim=True),
                ),
                dim=-1,
            )
            graph = Graph(
                mesh_nodes=reference_pos,
                senders=senders,
                receivers=receivers,
                node_features=torch.cat(
                    (node_type, velocity_prev, acceleration_prev, trac_feat),
                    dim=-1,
                ),
                edge_features=edge_features,
            )

            with torch.no_grad():
                u = mgn(graph=graph).to(dtype=dtype)

            if u.shape != u_prev.shape:
                raise ValueError(
                    f"Model prediction has shape {tuple(u.shape)}; "
                    f"expected {tuple(u_prev.shape)}."
                )

            u_tt = (u - 2.0 * u_prev + u_prev2) / (dt**2)
            displacement_xyz = torch.zeros_like(mesh.nodes)
            displacement_xyz[:, :output_dim] = u
            positions_xyz = mesh.nodes + displacement_xyz
            payload = np.asarray(
                positions_xyz.detach().cpu().numpy(),
                dtype="<f4",
            ).reshape(-1, 3).tobytes()
            udp_socket.sendto(payload, (UNITY_IP, UNITY_UDP_PORT))

            u_prev2 = u_prev
            u_prev = u
            velocity_prev = (u_prev - u_prev2) / dt
            acceleration_prev = u_tt
            world_pos = positions_xyz.detach()
            frame_index += 1
            time.sleep(0.016)


if __name__ == "__main__":
    stream_predictions()
