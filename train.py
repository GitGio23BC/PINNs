import torch
import numpy as np
import yaml
import logging
from pathlib import Path
from torch.optim import Adam

# Import dei moduli della mesh e del grafo (Cartelle src.mesh e src.graph)
from src.mesh.mesh import create_mesh
from src.graph.graph import create_graph

# Import dei moduli GNN (Cartella src.models)
from src.models.encoder import Encoder
from src.models.processor import MeshGraphNetProcessor
from src.models.decoder import Decoder
from src.models.backbones import MLP

# Import dei modelli PINN e loss (Cartella src.physics)
from src.physics.operators import haslach_constitutive_evolution_2D
from src.physics.loss import bc_loss, r_loss, mse

# Configurazione del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridDisplacementModel(torch.nn.Module):
    """
    Modello Coordinatore Ibrido: PINN (continua) + MeshGraphNet (discreto).
    Risolve il problema del campionamento valutando la PINN direttamente sui nodi materiali X.
    """
    def __init__(self, pinn_backbone, mgn_encoder, mgn_processor, mgn_decoder):
        super().__init__()
        self.pinn = pinn_backbone
        self.encoder = mgn_encoder
        self.processor = mgn_processor
        self.decoder = mgn_decoder

    def forward(self, X, senders, receivers, node_features, edge_features):
        # 1. Spostamento Continuo della PINN valutato sui nodi materiali X della mesh
        u_pinn = self.pinn(X)
        
        # 2. Aggiorniamo le feature del nodo includendo lo stato fisico predetto dalla PINN
        node_features_hybrid = torch.cat([node_features, u_pinn], dim=-1)
        
        # 3. Flusso GNN: Encode -> Process (Message Passing) -> Decode
        latent_nodes, latent_edges = self.encoder(node_features_hybrid, edge_features)
        updated_nodes, _ = self.processor(latent_nodes, latent_edges, senders, receivers)
        delta_u_mgn = self.decoder(updated_nodes)
        
        # 4. Spostamento ibrido finale coordinato
        u_final = u_pinn + delta_u_mgn
        
        return u_pinn, delta_u_mgn, u_final

def train_hybrid_model():
    # 1. Generazione della mesh e della connettività del grafo
    mesh = create_mesh(width=1.0, height=1.0, nx=5, ny=5)
    graph = create_graph(mesh)
    
    # IMPORTANTE: attiviamo requires_grad=True per abilitare l'autodifferenziazione spaziale
    X_nodes = torch.tensor(mesh.nodes, dtype=torch.float32).requires_grad_(True)
    
    senders = torch.tensor(graph.senders, dtype=torch.long)
    receivers = torch.tensor(graph.receivers, dtype=torch.long)
    
    # Feature del grafo di input
    node_features = torch.randn(X_nodes.shape[0], 1) # Feature strutturali di base
    edge_features = torch.randn(senders.shape[0], 6) # Distanze e orientamenti degli archi
    
    # Spostamento Ground Truth FEM per l'addestramento data-driven
    # Sganciato (.detach()) dal grafo computazionale di X_nodes per evitare il RuntimeError su epoche successive
    u_ground_truth = (torch.sin(X_nodes * np.pi) * 0.1).detach()
    
    # 2. Inizializzazione dei sottomodelli
    # PINN Backbone (Input: coordinate 2D X, Output: spostamenti 2D)
    pinn_backbone = MLP(in_dim=2, hidden_layers=4, hidden_dim=64, out_dim=2)
    
    # MeshGraphNet (La feature del nodo in ingresso è pari a: node_features(1) + u_pinn(2) = 3)
    mgn_encoder = Encoder(node_input_dim=3, edge_input_dim=6, latent_dim=32, hidden_dim=64)
    
    # Processor: accetta node_feature_dim=32 e edge_feature_dim=32 e hidden_dim=64
    mgn_processor = MeshGraphNetProcessor(node_feature_dim=32, edge_feature_dim=32, hidden_dim=64)
    
    # Il Decoder accetta latent_dim=64 (hidden_dim del Processor)
    mgn_decoder = Decoder(latent_dim=64, hidden_dim=64, output_dim=2)
    
    # Modello Ibrido Coordinatore
    hybrid_model = HybridDisplacementModel(pinn_backbone, mgn_encoder, mgn_processor, mgn_decoder)
    
    # Ottimizzatore congiunto
    optimizer = Adam(hybrid_model.parameters(), lr=1e-3)
    
    logger.info("Inizio addestramento congiunto Hybrid PINN + GNN...")
    
    for epoch in range(100):
        optimizer.zero_grad()
        
        # Forward pass ibrido
        u_pinn, delta_u_mgn, u_final = hybrid_model(
            X_nodes, senders, receivers, node_features, edge_features
        )
        
        # 3. Calcolo delle Perdite
        loss_data = mse(u_final, u_ground_truth)
        
        # Loss Fisica (Modello Viscoelastico di Haslach calcolato discretamente sugli elementi della mesh)
        _, _, physical_residuals = haslach_constitutive_evolution_2D(
            pinn_or_hybrid_displacement=u_final,
            X_ref=X_nodes,
            elements=mesh.elements,
            k=5.0, c1=4.0, c2=4.0, c3=8.0, c=0.88,
            b=torch.zeros(2),
            mode="discrete"
        )
        loss_physics = r_loss(physical_residuals)
        
        # Ponderazione con parametro di regolarizzazione lambda
        lambda_F = 0.1
        total_loss = loss_data + lambda_F * loss_physics
        
        # Backward pass unificato
        total_loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            logger.info(f"Epoca {epoch:03d} | Loss Totale: {total_loss.item():.6f} | Loss Dati: {loss_data.item():.6f} | Loss Fisica: {loss_physics.item():.6f}")

if __name__ == "__main__":
    train_hybrid_model()