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
    from src.physics.loss import mse, r_loss
except ImportError as e:
    print(f"Errore di importazione: {e}")
    print("Assicurati di lanciare lo script dalla cartella principale (root) della tua repo.")
    print("Esempio: python test_local_mesh_v6.py")
    exit(1)

# ==============================================================================
# OPERATORI FISICI DI HASLACH 2D CON STABILIZZAZIONE DELL'HESSIANO
# ==============================================================================
def haslach_constitutive_evolution_2D(
    u_hybrid: torch.Tensor,          # Spostamento ibrido incognito al passo corrente (u_pinn + delta_u_gnn)
    X_ref: torch.Tensor,             # Coordinate materiali di riferimento dei nodi (X)
    E_prev_voigt: torch.Tensor,      # Tensore di Green-Lagrange Voigt salvato al passo t_{n-1} (congelato!)
    elements: np.ndarray,            # Connettività degli elementi triangolari della mesh
    k_relax: float,                  # Modulo di rilassamento viscoelastico (k)
    c_isotropic: float,              # Contributo isotropo della matrice (c)
    k1_fiber: float,                 # Rigidità delle fibre (k1)
    k2_fiber: float,                 # Parametro esponenziale delle fibre (k2)
    beta_fiber: float,               # Angolo di orientamento delle fibre (beta) in radianti
    S_applied_voigt: torch.Tensor,   # Sforzo esterno applicato Voigt [S11, S22, S12] al passo corrente
    dt: float,                       # Passo temporale discreto (delta_t)
    reg_Hessian: float = 1e-2        # Parametro di stabilizzazione dell'Hessiano (neo-Hookean second-order stiffness)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calcola il residuo viscoelastico di Haslach 2D in modalità tempo-discretizzata (Seq2Seq).
    Risolve in modo differenziabile il marching-in-time su elementi triangolari.
    Implementa una stabilizzazione dell'Hessiano H per impedire la singolarità strutturale (det_H = 0).
    """
    from src.mechanics.deformation import compute_element_deformation_gradients

    # 1. Coordinate deformate correnti: x = X + u_hybrid
    x_deformed = X_ref + u_hybrid
    
    # 2. Calcolo discreto del gradiente di deformazione F per ciascun elemento della mesh
    F_discreto = compute_element_deformation_gradients(
        X_ref.detach().cpu().numpy(),
        x_deformed.detach().cpu().numpy(),
        elements
    )
    F_tensor = torch.tensor(F_discreto, dtype=torch.float32, device=u_hybrid.device)
    
    # 3. Calcolo del Tensore di Green-Lagrange corrente E^n = 0.5 * (F^T * F - I) (3D per incompressibilità)
    num_elements = F_tensor.shape[0]
    I = torch.eye(3, device=u_hybrid.device).unsqueeze(0).repeat(num_elements, 1, 1)
    E_tensor = 0.5 * (torch.bmm(F_tensor.transpose(1, 2), F_tensor) - I)
    
    # Componenti principali diagonalizzate (E22, E33)
    E2 = E_tensor[:, 0, 0]
    E3 = E_tensor[:, 1, 1]
    
    # Costruiamo il tensore in notazione Voigt [num_elementi, 3] -> [E11, E22, 2*E12]
    E_current_voigt = torch.zeros(num_elements, 3, device=u_hybrid.device)
    E_current_voigt[:, 0] = E2
    E_current_voigt[:, 1] = E3
    E_current_voigt[:, 2] = 2.0 * E_tensor[:, 0, 1]
    
    # 4. Velocità di deformazione discreta nel tempo: dot{E} = (E^n - E^{n-1}) / dt
    E_prev2 = E_prev_voigt[:, 0]
    E_prev3 = E_prev_voigt[:, 1]
    
    E_dot2 = (E2 - E_prev2) / dt
    E_dot3 = (E3 - E_prev3) / dt
    E_dot = torch.stack([E_dot2, E_dot3], dim=-1) # [num_elementi, 2]
    
    # 5. Calcolo dei parametri delle fibre anisotropiche HGO (Passo 515 del paper)
    c1 = 4.0 * k2_fiber * (np.cos(beta_fiber) ** 4)
    c2 = 4.0 * k2_fiber * (np.sin(beta_fiber) ** 4)
    c3 = 8.0 * k2_fiber * (np.cos(beta_fiber) ** 2) * (np.sin(beta_fiber) ** 2)
    
    exponent = c1 * (E2 ** 2) + c2 * (E3 ** 2) + c3 * E2 * E3
    exp_term = torch.exp(exponent)
    
    # 6. Derivate prime dell'energia di deformazione iperelastica W (dW/dE2, dW/dE3)
    dW_dE2 = c_isotropic + k1_fiber * (2.0 * c1 * E2 + c3 * E3) * exp_term
    dW_dE3 = c_isotropic + k1_fiber * (2.0 * c2 * E3 + c3 * E2) * exp_term
    
    # 7. Derivate seconde dell'energia di deformazione (Hessiano)
    H11 = k1_fiber * (2.0 * c1 + (2.0 * c1 * E2 + c3 * E3) ** 2) * exp_term
    H22 = k1_fiber * (2.0 * c2 + (2.0 * c2 * E3 + c3 * E2) ** 2) * exp_term
    H12 = k1_fiber * (c3 + (2.0 * c1 * E2 + c3 * E3) * (2.0 * c2 * E3 + c3 * E2)) * exp_term
    
    # STABILIZZAZIONE FISICA E NUMERICA DELL'HESSIANO
    # A causa della simmetria delle fibre nel piano 2D, l'Hessiano delle fibre è strutturalmente singolare (det=0).
    # L'aggiunta di un termine reg_Hessian agisce come contributo di rigidezza del secondo ordine della matrice isotropa (neo-Hookean).
    H11 = H11 + reg_Hessian
    H22 = H22 + reg_Hessian
    
    # Calcolo del determinante stabilizzato
    det_H = H11 * H22 - (H12 ** 2)
    
    # Inversione differenziabile della matrice 2x2
    invH11 = H22 / det_H
    invH22 = H11 / det_H
    invH12 = -H12 / det_H
    
    # Sforzo esterno applicato (componenti principali Sh e Sz)
    S2 = S_applied_voigt[:, 0]
    S3 = S_applied_voigt[:, 1]
    
    diff2 = S2 - dW_dE2
    diff3 = S3 - dW_dE3
    
    # dot{E}_target = k_relax * H^-1 * (S_applied - dW/dE)
    E_dot_target2 = k_relax * (invH11 * diff2 + invH12 * diff3)
    E_dot_target3 = k_relax * (invH12 * diff2 + invH22 * diff3)
    E_dot_target = torch.stack([E_dot_target2, E_dot_target3], dim=-1)
    
    # Residuo di Haslach
    residual = E_dot - E_dot_target
    
    return E_current_voigt, residual

# ==============================================================================
# MODELLO COORDINATORE IBRIDO
# ==============================================================================
class HybridDisplacementModel(torch.nn.Module):
    def __init__(self, pinn_backbone, mgn_encoder, mgn_processor, mgn_decoder):
        super().__init__()
        self.pinn = pinn_backbone
        self.encoder = mgn_encoder
        self.processor = mgn_processor
        self.decoder = mgn_decoder

    def forward(self, X, senders, receivers, node_features, edge_features):
        u_pinn = self.pinn(X)
        node_features_hybrid = torch.cat([node_features, u_pinn], dim=-1)
        latent_nodes, latent_edges = self.encoder(node_features_hybrid, edge_features)
        updated_nodes, _ = self.processor(latent_nodes, latent_edges, senders, receivers)
        delta_u_mgn = self.decoder(updated_nodes)
        u_final = u_pinn + delta_u_mgn
        return u_pinn, delta_u_mgn, u_final

# ==============================================================================
# FUNZIONE DI TEST
# ==============================================================================
def run_local_interactive_test():
    print("=" * 80)
    print("          TEST INTERATTIVO LOCALE DI DEFORMAZIONE DELLA MESH (v6 - STABILIZZATO)")
    print("=" * 80)
    
    print("\n1. Creazione della mesh quadrata regolare 5x5 tramite src.mesh.mesh...")
    mesh = create_mesh(width=1.0, height=1.0, nx=5, ny=5)
    graph = create_graph(mesh)
    
    X_nodes = torch.tensor(mesh.nodes, dtype=torch.float32).requires_grad_(True)
    senders = torch.tensor(graph.senders, dtype=torch.long)
    receivers = torch.tensor(graph.receivers, dtype=torch.long)
    
    node_features = torch.randn(X_nodes.shape[0], 1)
    edge_features = torch.randn(senders.shape[0], 6)
    
    # 2. Deformazione di riferimento fissa (Ground Truth staccata dal gradiente)
    u_ground_truth = torch.zeros_like(X_nodes)
    u_ground_truth[:, 0] = 0.15 * X_nodes[:, 0]
    u_ground_truth[:, 1] = -0.10 * torch.sin(X_nodes[:, 0] * np.pi)
    u_ground_truth = u_ground_truth.detach()
    
    print("2. Inizializzazione dei modelli (MLP Backbone + Encoder + Processor + Decoder)...")
    pinn_backbone = MLP(in_dim=2, hidden_layers=3, hidden_dim=32, out_dim=2)
    mgn_encoder = Encoder(node_input_dim=3, edge_input_dim=6, latent_dim=32, hidden_dim=64)
    mgn_processor = MeshGraphNetProcessor(node_feature_dim=32, edge_feature_dim=32, hidden_dim=64)
    mgn_decoder = Decoder(latent_dim=64, hidden_dim=64, output_dim=2)
    
    hybrid_model = HybridDisplacementModel(pinn_backbone, mgn_encoder, mgn_processor, mgn_decoder)
    optimizer = torch.optim.Adam(hybrid_model.parameters(), lr=0.01)
    
    # 3. Parametri del Ciclo Temporale Seq2Seq (Marching-in-time)
    print("\n3. Avvio del ciclo temporale Seq2Seq (marching-in-time)...")
    timesteps = 5
    epochs_per_step = 100
    dt = 0.05
    k_relax = 5.0
    c_isotropic = 0.88
    k1_fiber = 2.0
    k2_fiber = 4.0
    beta_fiber = np.pi / 4.0 # 45 gradi
    
    num_elements = len(mesh.elements)
    
    # Stato della deformazione precedente congelato
    E_prev_voigt = torch.zeros(num_elements, 3, dtype=torch.float32)
    
    losses_total = []
    losses_data = []
    losses_physics = []
    
    pinn_norms = []
    gnn_norms = []
    
    for t_step in range(1, timesteps + 1):
        t_current = t_step * dt
        print(f"\n   --- PASSO TEMPORALE {t_step}/{timesteps} (t = {t_current:.2f}s) ---")
        
        # Carico pulsante nel tempo
        S_applied_val = 1.0 * np.sin(2.0 * np.pi * t_current)
        S_applied_voigt = torch.zeros(num_elements, 3, dtype=torch.float32)
        S_applied_voigt[:, 0] = S_applied_val
        S_applied_voigt[:, 1] = 0.5 * S_applied_val
        
        # Stato target temporaneo staccato
        u_target = u_ground_truth * np.sin(2.0 * np.pi * t_current)
        u_target = u_target.detach()
        
        for epoch in range(1, epochs_per_step + 1):
            optimizer.zero_grad()
            
            # Forward pass ibrido
            u_pinn, delta_u_mgn, u_final = hybrid_model(
                X_nodes, senders, receivers, node_features, edge_features
            )
            
            # Loss sui Dati
            loss_d = mse(u_final, u_target)
            
            # Loss Fisica (stabilizzata internamente tramite reg_Hessian)
            E_current_voigt, physical_residuals = haslach_constitutive_evolution_2D(
                u_hybrid=u_final,
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
                reg_Hessian=1e-1 # Regolarizzazione dell'Hessiano contro la singolarità
            )
            loss_p = r_loss(physical_residuals)
            
            # Loss totale
            total_loss = loss_d + 0.1 * loss_p
            
            total_loss.backward()
            optimizer.step()
            
            # Salvataggio metriche
            losses_total.append(total_loss.item())
            losses_data.append(loss_d.item())
            losses_physics.append(loss_p.item())
            
            if epoch % 5 == 0 or epoch == 1:
                print(f"      Epoca {epoch:02d}/{epochs_per_step} | Loss Totale: {total_loss.item():.6f} | Loss Dati: {loss_d.item():.6f} | Loss Fisica: {loss_p.item():.6f}")
                
        # Congelamento dello stato di deformazione per il passo temporale successivo (Seq2Seq)
        E_prev_voigt = E_current_voigt.detach()
        
        # Salvataggio delle norme medie per monitoraggio NN
        pinn_norms.append(torch.mean(torch.linalg.norm(u_pinn, dim=1)).item())
        gnn_norms.append(torch.mean(torch.linalg.norm(delta_u_mgn, dim=1)).item())

    # Estrazione risultati finali per il plot
    with torch.no_grad():
        u_pinn_val, delta_u_mgn_val, u_final_val = hybrid_model(
            X_nodes, senders, receivers, node_features, edge_features
        )
        u_final_np = u_final_val.numpy()
        u_pinn_np = u_pinn_val.numpy()
        u_mgn_np = delta_u_mgn_val.numpy()
        
    nodes_deformed = mesh.nodes + u_final_np
    
    # ==============================================================================
    # GENERAZIONE DEI PLOT COMPARATIVI
    # ==============================================================================
    print("\n4. Generazione del grafico dell'evoluzione temporale in corso...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Mesh originale vs Mesh Deformata
    for element in mesh.elements:
        pts = mesh.nodes[element]
        ax1.fill(pts[:, 0], pts[:, 1], edgecolor='gray', facecolor='none', linestyle='--', alpha=0.5)
    for element in mesh.elements:
        pts = nodes_deformed[element]
        ax1.fill(pts[:, 0], pts[:, 1], edgecolor='blue', facecolor='lightblue', alpha=0.3)
        
    ax1.scatter(nodes_deformed[:, 0], nodes_deformed[:, 1], color='blue', label='Deformata (t=0.25s)', zorder=5)
    ax1.scatter(mesh.nodes[:, 0], mesh.nodes[:, 1], color='red', marker='x', label='Originale (t=0.00s)', zorder=4)
    ax1.set_title("Evoluzione Spaziale della Mesh")
    ax1.set_xlabel("Coordinata X")
    ax1.set_ylabel("Coordinata Y")
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Plot 2: Convergence della loss
    epochs_total = len(losses_total)
    ax2.plot(range(1, epochs_total + 1), losses_total, label="Loss Totale", color='purple', linewidth=2)
    ax2.plot(range(1, epochs_total + 1), losses_data, label="Loss Dati", color='green', linestyle='--')
    ax2.plot(range(1, epochs_total + 1), losses_physics, label="Loss Fisica (Haslach)", color='orange', linestyle=':')
    ax2.set_yscale("log")
    ax2.set_title("Andamento della Loss (Seq2Seq)")
    ax2.set_xlabel("Epoche Complessive")
    ax2.set_ylabel("Valore della Loss (Scala Log)")
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # Plot 3: Contributo PINN vs GNN nel tempo
    ax3.plot(range(1, timesteps + 1), pinn_norms, marker='o', label="Spostamento PINN", color='blue', linewidth=2)
    ax3.plot(range(1, timesteps + 1), gnn_norms, marker='s', label="Correzione GNN", color='cyan', linewidth=2)
    ax3.set_title("Contributi delle Reti Neurali nel Tempo")
    ax3.set_xlabel("Passo Temporale")
    ax3.set_ylabel("Norma Media dello Spostamento [m]")
    ax3.legend()
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    output_image = "test_mesh_temporal_evolution.png"
    plt.savefig(output_image, dpi=150)
    plt.close()
    
    print(f"\n[SUCCESSO] Grafico temporale salvato come: '{output_image}'")
    print("\n--- DATI DI OUTPUT ---")
    print(f"Spostamento combinato finale u_final (campione primi 3 nodi):\n{u_final_np[:3]}")
    print(f"Contributo medio PINN (norma): {pinn_norms[-1]:.5f}")
    print(f"Contributo medio correzione GNN (norma): {gnn_norms[-1]:.5f}")
    print("================================================================================")

if __name__ == "__main__":
    run_local_interactive_test()
