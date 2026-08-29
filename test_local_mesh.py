import torch
import numpy as np
import matplotlib.pyplot as plt

# Assicurati che VS Code riconosca il path 'src' se esegui dalla root del progetto.
try:
    from src.mesh.mesh import create_mesh
    from src.graph.graph import create_graph
    from src.models.encoder import Encoder
    from src.models.processor import MeshGraphNetProcessor
    from src.models.decoder import Decoder
    from src.models.backbones import MLP
    from src.physics.operators import haslach_constitutive_evolution_2D
    from src.physics.loss import mse, r_loss
except ImportError as e:
    print(f"Errore di importazione: {e}")
    print("Assicurati di lanciare lo script dalla cartella principale (root) della tua repo.")
    print("Esempio: python test_local_mesh_v4.py")
    exit(1)

class HybridDisplacementModel(torch.nn.Module):
    """
    Modello Coordinatore Ibrido: PINN (continua) + MeshGraphNet (discreta).
    Valuta la PINN continua direttamente sui nodi materiali X della mesh.
    """
    def __init__(self, pinn_backbone, mgn_encoder, mgn_processor, mgn_decoder):
        super().__init__()
        self.pinn = pinn_backbone
        self.encoder = mgn_encoder
        self.processor = mgn_processor
        self.decoder = mgn_decoder

    def forward(self, X, senders, receivers, node_features, edge_features):
        # 1. Spostamento continuo generato dalla PINN sui nodi materiali X
        u_pinn = self.pinn(X)
        
        # 2. Concatena le caratteristiche dei nodi con lo spostamento della PINN
        node_features_hybrid = torch.cat([node_features, u_pinn], dim=-1)
        
        # 3. Flusso MeshGraphNet: Encoder -> Processor (Message Passing) -> Decoder
        latent_nodes, latent_edges = self.encoder(node_features_hybrid, edge_features)
        updated_nodes, _ = self.processor(latent_nodes, latent_edges, senders, receivers)
        delta_u_mgn = self.decoder(updated_nodes)
        
        # 4. Spostamento combinato finale (Ibrido)
        u_final = u_pinn + delta_u_mgn
        
        return u_pinn, delta_u_mgn, u_final

def run_local_interactive_test():
    print("=" * 80)
    print("             TEST INTERATTIVO LOCALE DI DEFORMAZIONE DELLA MESH (v4 - RISOLTO)")
    print("=" * 80)
    
    # 1. Creazione della mesh quadrata regolare 5x5 tramite src.mesh.mesh
    # ================================================================================
    print("\n1. Creazione della mesh quadrata regolare 5x5 tramite src.mesh.mesh...")
    mesh = create_mesh(width=1.0, height=1.0, nx=5, ny=5)
    graph = create_graph(mesh)
    
    # Estraiamo i tensori e abilitiamo requires_grad_ per l'autograd continuo (PINN)
    X_nodes = torch.tensor(mesh.nodes, dtype=torch.float32).requires_grad_(True)
    senders = torch.tensor(graph.senders, dtype=torch.long)
    receivers = torch.tensor(graph.receivers, dtype=torch.long)
    
    # Generiamo feature strutturali per il grafo (es. 1 per nodo, 6 per arco)
    node_features = torch.randn(X_nodes.shape[0], 1)
    edge_features = torch.randn(senders.shape[0], 6)
    
    # 2. Ipotizziamo una deformazione di riferimento (allungamento + curvatura)
    # ================================================================================
    # CORREZIONE DI SICUREZZA: detacchiamo u_ground_truth dal grafo computazionale di X_nodes.
    # In quanto target statico (Ground Truth), non deve far propagare i gradienti indietro,
    # altrimenti PyTorch proverà a fare backward due volte sulla stessa porzione di grafo
    # sollevando il RuntimeError sui tensori liberati (freed tensors).
    u_ground_truth = torch.zeros_like(X_nodes)
    u_ground_truth[:, 0] = 0.15 * X_nodes[:, 0]                     # Allungamento del 15% su X
    u_ground_truth[:, 1] = -0.10 * torch.sin(X_nodes[:, 0] * np.pi) # Curvatura su Y
    u_ground_truth = u_ground_truth.detach()                         # Sgancia dal grafo computazionale!
    
    # 3. Inizializzazione dei modelli con le dimensioni corrette
    # ================================================================================
    print("2. Inizializzazione dei modelli (MLP Backbone + Encoder + Processor + Decoder)...")
    # PINN MLP: prende coordinate (2D) e restituisce spostamenti (2D)
    pinn_backbone = MLP(in_dim=2, hidden_layers=3, hidden_dim=32, out_dim=2)
    
    # MeshGraphNet: l'input del nodo è features (1D) + u_pinn (2D) = 3D
    mgn_encoder = Encoder(node_input_dim=3, edge_input_dim=6, latent_dim=32, hidden_dim=64)
    
    # Processor: accetta node_feature_dim=32 (da latent_dim dell'encoder) e edge_feature_dim=32
    # e hidden_dim=64. L'output dei nodi ha dimensione hidden_dim = 64!
    mgn_processor = MeshGraphNetProcessor(node_feature_dim=32, edge_feature_dim=32, hidden_dim=64)
    
    # Il Decoder accetta latent_dim=64, perché l'output di MeshGraphNetProcessor ha dimensione hidden_dim = 64.
    mgn_decoder = Decoder(latent_dim=64, hidden_dim=64, output_dim=2)
    
    # Modello coordinatore ibrido
    hybrid_model = HybridDisplacementModel(pinn_backbone, mgn_encoder, mgn_processor, mgn_decoder)
    
    # Ottimizzatore Adam
    optimizer = torch.optim.Adam(hybrid_model.parameters(), lr=0.01)
    
    # 4. Eseguiamo un ciclo di addestramento rapido (es. 20 epoche)
    # ================================================================================
    print("\n3. Avvio di un micro-ciclo di ottimizzazione locale (100 epoche)...")
    epochs = 100
    losses = []
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        # Forward pass
        u_pinn, delta_u_mgn, u_final = hybrid_model(
            X_nodes, senders, receivers, node_features, edge_features
        )
        
        # Loss sui dati: lo spostamento totale deve approssimare la deformazione ipotizzata
        loss_data = mse(u_final, u_ground_truth)
        
        # Loss fisica: residui di Haslach (valutati discretamente sugli elementi della tua mesh)
        _, _, physical_residuals = haslach_constitutive_evolution_2D(
            pinn_or_hybrid_displacement=u_final,
            X_ref=X_nodes,
            elements=mesh.elements,
            k=5.0, c1=1.0, c2=1.0, c3=2.0, c=0.5,
            b=torch.zeros(2),
            mode="discrete"
        )
        loss_physics = r_loss(physical_residuals)
        
        total_loss = loss_data + 0.1 * loss_physics
        losses.append(total_loss.item())
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"   Epoca {epoch:02d}/{epochs} | Loss Totale: {total_loss.item():.6f} | Loss Dati: {loss_data.item():.6f} | Loss Fisica: {loss_physics.item():.6f}")

    # Estraiamo i risultati finali per il plotting
    with torch.no_grad():
        _, _, u_final_val = hybrid_model(X_nodes, senders, receivers, node_features, edge_features)
        u_final_np = u_final_val.numpy()
        u_pinn_np = u_pinn.numpy()
        u_mgn_np = delta_u_mgn.numpy()
    
    # Posizioni deformate finali dei nodi: x = X + u
    nodes_deformed = mesh.nodes + u_final_np
    
    # 5. Generazione e salvataggio dei grafici (Immagine)
    # =============================================================================== =
    print("\n4. Generazione del grafico comparativo in corso...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Grafico 1: Mesh originale indeformata vs Mesh Deformata
    # Disegniamo la mesh originale (linee grigie sottili)
    for element in mesh.elements:
        pts = mesh.nodes[element]
        ax1.fill(pts[:, 0], pts[:, 1], edgecolor='gray', facecolor='none', linestyle='--', alpha=0.5)
    
    # Disegniamo la mesh deformata (linee blu, nodi blu)
    for element in mesh.elements:
        pts = nodes_deformed[element]
        ax1.fill(pts[:, 0], pts[:, 1], edgecolor='blue', facecolor='lightblue', alpha=0.3)
    ax1.scatter(nodes_deformed[:, 0], nodes_deformed[:, 1], color='blue', label='Deformata', zorder=5)
    ax1.scatter(mesh.nodes[:, 0], mesh.nodes[:, 1], color='red', marker='x', label='Originale', zorder=4)
    
    ax1.set_title("Deformazione della Mesh Quadrata (PINN + MGN)")
    ax1.set_xlabel("Coordinata X")
    ax1.set_ylabel("Coordinata Y")
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Grafico 2: Andamento della perdita
    ax2.plot(range(1, epochs + 1), losses, marker='o', color='purple', linewidth=2)
    ax2.set_title("Andamento della Loss in 100 Epoche")
    ax2.set_xlabel("Epoca")
    ax2.set_ylabel("Loss Totale")
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    output_img_name = "test_mesh_deformation.png"
    plt.savefig(output_img_name, dpi=150)
    plt.close()
    
    # 6. Output dei Dati Numerici a schermo
    # =============================================================================== =
    print(f"\n[SUCCESSO] Immagine salvata localmente come: '{output_img_name}'")
    print("\n--- DATI DI OUTPUT ---")
    print(f"Spianamento nodi originali (campione primi 3 nodi):\n{mesh.nodes[:3]}")
    print(f"Spostamento combinato finale u_final (campione primi 3 nodi):\n{u_final_np[:3]}")
    print(f"Posizioni dei nodi deformati x = X + u (campione primi 3 nodi):\n{nodes_deformed[:3]}")
    print(f"Contributo medio della PINN continua (norma): {np.mean(np.linalg.norm(u_pinn_np, axis=1)):.5f}")
    print(f"Contributo medio della correzione GNN (norma): {np.mean(np.linalg.norm(u_mgn_np, axis=1)):.5f}")
    print("================================================================================")

if __name__ == "__main__":
    run_local_interactive_test()
