import torch
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. IMPORT DEI TUOI MODULI LOCALI
# ==============================================================================
try:
    # Mesh e Graph: caricati da src.mesh e src.graph
    try:
        from src.mesh.mesh import create_mesh
    except ImportError:
        from src.mesh import create_mesh

    try:
        from src.graph.graph import create_graph
    except ImportError:
        from src.graph import create_graph
    
    # Modelli (MLP, Encoder, Processor, Decoder) caricati da src.models
    from src.models.encoder import Encoder
    try:
        from src.models.processor import MeshGraphNetProcessor
    except ImportError:
        from src.models.mesh_graph_net import MeshGraphNetProcessor
    from src.models.decoder import Decoder
    from src.models.backbones import MLP
    
    # Operatori e loss caricati da src.physics
    from src.physics.operators import haslach_constitutive_evolution_2D
    from src.physics.loss import mse, r_loss

except ImportError as e:
    print(f"\n[ERRORE IMPORT] Non riesco a trovare i moduli: {e}")
    print("Assicurati di eseguire lo script dalla root della tua repository locale.")
    print("La cartella 'src' deve essere visibile nello stesso percorso in cui esegui questo file.")
    exit(1)


# ==============================================================================
# 2. DEFINIZIONE DEL MODELLO IBRIDO COORDINATORE
# ==============================================================================
class HybridDisplacementModel(torch.nn.Module):
    def __init__(self, pinn_backbone, mgn_encoder, mgn_processor, mgn_decoder):
        super().__init__()
        self.pinn = pinn_backbone
        self.encoder = mgn_encoder
        self.processor = mgn_processor
        self.decoder = mgn_decoder

    def forward(self, X, senders, receivers, node_features, edge_features):
        # Spostamento continuo della PINN valutato sui nodi materiali X della mesh
        u_pinn = self.pinn(X)
        
        # Concatena le caratteristiche strutturali con lo spostamento PINN
        node_features_hybrid = torch.cat([node_features, u_pinn], dim=-1)
        
        # Flusso GNN: Encode -> Process (Message Passing) -> Decode
        latent_nodes, latent_edges = self.encoder(node_features_hybrid, edge_features)
        updated_nodes, _ = self.processor(latent_nodes, latent_edges, senders, receivers)
        delta_u_mgn = self.decoder(updated_nodes)
        
        # Spostamento ibrido finale coordinato
        u_final = u_pinn + delta_u_mgn
        
        return u_pinn, delta_u_mgn, u_final


# ==============================================================================
# 3. CICLO DI TEST INTERATTIVO
# ==============================================================================
def run_local_visual_test():
    print("=" * 80)
    print("          AVVIO TEST LOCALE DI DEFORMAZIONE ED OTTIMIZZAZIONE (VS CODE)")
    print("=" * 80)

    # 1. Creazione della mesh e del grafo tramite i tuoi script src.mesh e src.graph
    print("\n[INFO] 1. Creazione della mesh quadrata regolare 5x5...")
    mesh = create_mesh(width=1.0, height=1.0, nx=5, ny=5)
    graph = create_graph(mesh)
    
    # Abilitiamo requires_grad=True per l'autodifferenziazione (PINN)
    X_nodes = torch.tensor(mesh.nodes, dtype=torch.float32).requires_grad_(True)
    senders = torch.tensor(graph.senders, dtype=torch.long)
    receivers = torch.tensor(graph.receivers, dtype=torch.long)
    
    # CORREZIONE ERRORE randn(): usiamo X_nodes.shape[0] per ottenere il numero di nodi (36)
    # invece di passare l'intero torch.Size([36, 2]) che causava il TypeError!
    node_features = torch.randn(X_nodes.shape[0], 1)
    edge_features = torch.randn(senders.shape[0], 6)
    
    # 2. Definiamo uno spostamento ideale Ground Truth (Target costante staccato dal gradiente)
    u_ground_truth = torch.zeros_like(X_nodes)
    u_ground_truth[:, 0] = 0.15 * X_nodes[:, 0]                     # Allungamento del 15% su X
    u_ground_truth[:, 1] = -0.10 * torch.sin(X_nodes[:, 0] * np.pi) # Curvatura sinusoidale su Y
    u_ground_truth = u_ground_truth.detach()                         # <--- Correzione Autograd!
    
    # 3. Inizializzazione dei sottomodelli con le dimensioni coerenti (latent_dim=32, hidden_dim=64)
    print("[INFO] 2. Inizializzazione dei tuoi sottomodelli PyTorch...")
    pinn_backbone = MLP(in_dim=2, hidden_layers=3, hidden_dim=32, out_dim=2)
    mgn_encoder = Encoder(node_input_dim=3, edge_input_dim=6, latent_dim=32, hidden_dim=64)
    mgn_processor = MeshGraphNetProcessor(node_feature_dim=32, edge_feature_dim=32, hidden_dim=64)
    mgn_decoder = Decoder(latent_dim=64, hidden_dim=64, output_dim=2) # Decoder riceve 64 in uscita dal processor
    
    hybrid_model = HybridDisplacementModel(pinn_backbone, mgn_encoder, mgn_processor, mgn_decoder)
    optimizer = torch.optim.Adam(hybrid_model.parameters(), lr=0.01)
    
    # 4. Ciclo di addestramento rapido (20 epoche)
    print("\n[INFO] 3. Esecuzione dell'ottimizzazione congiunta (20 epoche)...")
    epochs = 20
    losses = []
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        # Forward pass ibrido
        u_pinn, delta_u_mgn, u_final = hybrid_model(
            X_nodes, senders, receivers, node_features, edge_features
        )
        
        # Loss Dati (confronto spostamento ibrido totale con la deformazione target)
        loss_data = mse(u_final, u_ground_truth)
        
        # Loss Fisica (modello viscoelastico discreto di Haslach sugli elementi della tua mesh)
        _, _, physical_residuals = haslach_constitutive_evolution_2D(
            pinn_or_hybrid_displacement=u_final,
            X_ref=X_nodes,
            elements=mesh.elements,
            k=5.0, c1=1.0, c2=1.0, c3=2.0, c=0.5,
            b=torch.zeros(2),
            mode="discrete"
        )
        loss_physics = r_loss(physical_residuals)
        
        # Calcolo perdita totale
        total_loss = loss_data + 0.1 * loss_physics
        losses.append(total_loss.item())
        
        total_loss.backward()
        optimizer.step()
        
        print(f"   Epoca {epoch:02d}/{epochs} | Loss Totale: {total_loss.item():.6f} | Loss Dati: {loss_data.item():.6f} | Loss Fisica: {loss_physics.item():.6f}")

    # Estrazione dei risultati numerici finali
    with torch.no_grad():
        _, _, u_final_val = hybrid_model(X_nodes, senders, receivers, node_features, edge_features)
        u_final_np = u_final_val.numpy()
        u_pinn_np = u_pinn.numpy()
        u_mgn_np = delta_u_mgn.numpy()
    
    # Calcolo posizioni deformate finali dei nodi per il plot
    nodes_deformed = mesh.nodes + u_final_np
    
    # ==============================================================================
    # 4. GENERAZIONE DEL PLOT GRAFICO (USANDO MATPLOTLIB)
    # ==============================================================================
    print("\n[INFO] 4. Generazione del grafico comparativo...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Mesh originale vs Mesh Deformata
    # Disegniamo la mesh originale triangolo per triangolo (tratteggio grigio)
    for element in mesh.elements:
        pts = mesh.nodes[element]
        ax1.fill(pts[:, 0], pts[:, 1], edgecolor='gray', facecolor='none', linestyle='--', alpha=0.5)
    
    # Disegniamo la mesh deformata triangolo per triangolo (linea continua blu e riempimento azzurro)
    for element in mesh.elements:
        pts = nodes_deformed[element]
        ax1.fill(pts[:, 0], pts[:, 1], edgecolor='blue', facecolor='lightblue', alpha=0.3)
        
    ax1.scatter(nodes_deformed[:, 0], nodes_deformed[:, 1], color='blue', label='Mesh Deformata', zorder=5)
    ax1.scatter(mesh.nodes[:, 0], mesh.nodes[:, 1], color='red', marker='x', label='Nodi Originali (X)', zorder=4)
    
    ax1.set_title("Deformazione della Mesh Quadrata (PINN + GNN)")
    ax1.set_xlabel("Coordinata X")
    ax1.set_ylabel("Coordinata Y")
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Plot 2: Andamento della perdita di addestramento
    ax2.plot(range(1, epochs + 1), losses, marker='o', color='purple', linewidth=2)
    ax2.set_title("Andamento della Loss in 20 Epoche")
    ax2.set_xlabel("Epoca")
    ax2.set_ylabel("Loss Totale")
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    
    # Salvataggio dell'immagine nella cartella locale
    output_image = "test_mesh_deformation_VSCODE.png"
    plt.savefig(output_image, dpi=150)
    plt.close()
    
    print(f"\n[SUCCESSO] Grafico salvato localmente come: '{output_image}'")
    print("\n--- DATI DI OUTPUT AGGREGATI ---")
    print(f"Spostamento combinato finale u_final (primi 3 nodi):\n{u_final_np[:3]}")
    print(f"Contributo medio PINN (norma): {np.mean(np.linalg.norm(u_pinn_np, axis=1)):.5f}")
    print(f"Contributo medio correzione GNN (norma): {np.mean(np.linalg.norm(u_mgn_np, axis=1)):.5f}")
    print("================================================================================")

if __name__ == "__main__":
    run_local_visual_test()
