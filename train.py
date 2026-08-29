import torch
import yaml
import logging
from pathlib import Path
from torch.optim import Adam

# Import dei moduli della mesh e del grafo (Cartelle src.mesh e src.graph)
from src.mesh.mesh import create_mesh
from src.graph.graph import create_graph

# Import dei moduli GNN (Cartella src.models)
from src.models.encoder import Encoder
from src.models.processor import Processor
from src.models.decoder import Decoder

# Import dei modelli PINN e loss (Cartella src.physics)
from src.physics.operators import haslach_constitutive_evolution_2D
from src.physics.loss import bc_loss, r_loss, mse

# Configurazione del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MeshGraphNet(torch.nn.Module):
    def __init__(self, mgn_encoder, mgn_processor, mgn_decoder):
        super().__init__()
        self.encoder = mgn_encoder
        self.processor = mgn_processor
        self.decoder = mgn_decoder

    def forward(self, senders, receivers, node_features, edge_features) -> torch.Tensor:
        latent_nodes, latent_edges = self.encoder(node_features, edge_features)
        updated_nodes, _ = self.processor(latent_nodes, latent_edges, senders, receivers)
        u_final = self.decoder(updated_nodes)
        return u_final

def train_hybrid_model():
    mesh = create_mesh(width=1.0, height=1.0, nx=5, ny=5)
    graph = create_graph(mesh)
    
    X_nodes = torch.tensor(mesh.nodes, dtype=torch.float32).requires_grad_(True)
    senders = torch.tensor(graph.senders, dtype=torch.long)
    receivers = torch.tensor(graph.receivers, dtype=torch.long)

    torch.manual_seed(42)
    node_features = torch.randn(X_nodes.shape[0], 1)
    edge_features = torch.randn(senders.shape[0], 6)
    
    mgn_encoder = Encoder(node_input_dim=3, edge_input_dim=6, latent_dim=32, hidden_dim=64)
    mgn_processor = Processor(node_feature_dim=32, edge_feature_dim=32, hidden_dim=64)
    mgn_decoder = Decoder(latent_dim=64, hidden_dim=64, output_dim=2)
    
    meshgraphnet_model = MeshGraphNet(mgn_encoder, mgn_processor, mgn_decoder)
    optimizer = Adam(meshgraphnet_model.parameters(), lr=1e-3)
    
    timesteps = 10
    epochs_per_step = 20
    dt = 0.05
    k_relax = 5.0
    c_isotropic = 0.88
    k1_fiber = 2.0
    k2_fiber = 4.0
    beta_fiber = torch.pi / 4.0
    
    num_elements = len(mesh.elements)
    E_prev_voigt = torch.zeros(num_elements, 3, dtype=torch.float32)
    
    logger.info("Inizio addestramento congiunto Hybrid PINN + GNN in modalità Sequence-to-Sequence (Hessiano Stabilizzato)...")
    
    for t_step in range(1, timesteps + 1):
        t_current = t_step * dt
        logger.info(f"\n--- PASSO TEMPORALE {t_step}/{timesteps} (t = {t_current:.3f}s) ---")
        
        S_applied_val = 1.0 * torch.sin(2.0 * torch.pi * t_current)
        
        u_ground_truth = torch.zeros_like(X_nodes)
        u_ground_truth[:, 0] = 0.15 * X_nodes[:, 0] * torch.sin(2.0 * torch.pi * t_current)
        u_ground_truth = u_ground_truth.detach()
        
        for epoch in range(1, epochs_per_step + 1):
            optimizer.zero_grad()
            
            u_final = meshgraphnet_model(
                X_nodes, senders, receivers, node_features, edge_features
            )
            
            loss_data = mse(u_final, u_ground_truth)
            
            E_current_voigt, physical_residuals = haslach_constitutive_evolution_2D(
                u_total=u_final,
                X_ref=X_nodes,
                E_prev_voigt=E_prev_voigt,
                elements=mesh.elements,
                k_relax=k_relax,
                c_isotropic=c_isotropic,
                k1_fiber=k1_fiber,
                k2_fiber=k2_fiber,
                beta_fiber=beta_fiber,
                S_applied_voigt=S_applied_voigt,
                dt=dt,
                reg_Hessian=1e-1,  # STABILIZZAZIONE DELL'HESSIANO
                mode="discrete"
            )
            loss_physics = r_loss(physical_residuals)
            
            total_loss = loss_data + 0.1 * loss_physics
            
            total_loss.backward()
            optimizer.step()
            
            if epoch % 5 == 0 or epoch == 1:
                logger.info(f"   Epoca {epoch:02d}/{epochs_per_step} | Loss Totale: {total_loss.item():.6f} | Loss Dati: {loss_data.item():.6f} | Loss Fisica: {loss_physics.item():.6f}")
        
        E_prev_voigt = E_current_voigt.detach()

if __name__ == "__main__":
    train_hybrid_model()