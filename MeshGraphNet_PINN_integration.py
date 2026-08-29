import torch
import torch.nn as nn

# ==============================================================================
# 1. ARCHITETTURA IBRIDA PINN + MESHGRAPHNET (TEMPLATE DI INTEGRAZIONE)
# ==============================================================================

class HybridDisplacementModel(nn.Module):
    """
    Modello Ibrido che combina la PINN continua e la rete neurale a grafo MeshGraphNet.
    Fornisce lo spostamento finale u_final = u_pinn + delta_u_mgn.
    """
    def __init__(self, pinn: nn.Module, mesh_graph_net: nn.Module):
        super().__init__()
        self.pinn = pinn
        self.mgn = mesh_graph_net

    def forward(self, X: torch.Tensor, senders: torch.Tensor, receivers: torch.Tensor, edge_features: torch.Tensor):
        # Assicuriamo il tracciamento del gradiente spaziale per la PINN
        X_with_grad = X.clone().detach().requires_grad_(True)
        
        # 1. Spostamento continuo generato dalla PINN
        u_pinn = self.pinn(X_with_grad)
        
        # 2. Costruzione delle feature dei nodi per MeshGraphNet
        # Combiniamo le coordinate iniziali X e lo spostamento continuo u_pinn
        # (Es. concatenazione [X, u_pinn] o altre feature di interesse)
        node_features = torch.cat([X_with_grad, u_pinn], dim=-1) # Dim: (N, 4)
        
        # 3. Spostamento correttivo generato da MeshGraphNet
        delta_u_mgn = self.mgn(node_features, edge_features, senders, receivers)
        
        # Spostamento finale ibrido
        u_final = u_pinn + delta_u_mgn
        
        return u_final, X_with_grad, u_pinn, delta_u_mgn

# ==============================================================================
# 2. TEST DI INTEGRAZIONE END-TO-END (MOCKING)
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("      TEST DI INTEGRAZIONE END-TO-END: HYBRID PINN + MESHGRAPHNET")
    print("=" * 80)

    torch.manual_seed(42)

    # 1. Configurazione della Mesh Simulata (4 nodi, 2D)
    # Coordinate materiali iniziali (X)
    X = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ], dtype=torch.float32)

    # Topologia del grafo (6 archi orientati)
    senders = torch.tensor([0, 1, 1, 2, 2, 3], dtype=torch.long)
    receivers = torch.tensor([1, 0, 2, 1, 3, 2], dtype=torch.long)
    
    # Feature fittizie degli archi (E, 6)
    edge_features = torch.randn(len(senders), 6)

    # 2. Inizializzazione della PINN (MLP Semplice)
    # Prende in input coordinate 2D (x, y) e predice spostamenti 2D (u_x, u_y)
    pinn_model = nn.Sequential(
        nn.Linear(2, 16),
        nn.Tanh(),
        nn.Linear(16, 2)
    )

    # 3. Mock di MeshGraphNet (Encoder -> Processor -> Decoder)
    # Per semplicità del test, implementiamo una classe compatta che simula il flusso
    class MockMeshGraphNet(nn.Module):
        def __init__(self, node_in_dim=4, edge_in_dim=6, latent_dim=32):
            super().__init__()
            self.node_encoder = nn.Linear(node_in_dim, latent_dim)
            self.edge_encoder = nn.Linear(edge_in_dim, latent_dim)
            # Processor (Message Passing ridotto)
            self.msg_mlp = nn.Linear(latent_dim * 2, latent_dim)
            # Decoder
            self.decoder = nn.Linear(latent_dim, 2)

        def forward(self, node_features, edge_features, senders, receivers):
            h_n = torch.tanh(self.node_encoder(node_features))
            h_e = torch.tanh(self.edge_encoder(edge_features))
            
            # Message passing minimale
            msg = torch.tanh(self.msg_mlp(torch.cat([h_n[senders], h_e], dim=-1)))
            agg_msg = torch.zeros_like(h_n)
            agg_msg.index_add_(0, receivers, msg)
            
            # Aggiornamento nodi e decodifica
            out = self.decoder(torch.tanh(h_n + agg_msg))
            return out

    mgn_model = MockMeshGraphNet(node_in_dim=4, edge_in_dim=6, latent_dim=32)

    # 4. Creazione del Modello Ibrido Integrato
    hybrid_model = HybridDisplacementModel(pinn_model, mgn_model)

    # 5. Forward Pass
    u_final, X_with_grad, u_pinn, delta_u_mgn = hybrid_model(X, senders, receivers, edge_features)

    print("\n[FORWARD PASS SUCCESSO]")
    print(f"Spostamento PINN (u_pinn):\n{u_pinn}")
    print(f"Correzione GNN (delta_u_mgn):\n{delta_u_mgn}")
    print(f"Spostamento finale combinato (u_final):\n{u_final}")

    # 6. Definizione di una Loss di Esempio (Residuo Fisico + Condizioni al Contorno)
    # Simuliamo una loss che richiede derivate spaziali sulla PINN (gradiente dello spostamento)
    # grad_u ha forma (N, 2, 2) rappresentando [du_x/dx, du_x/dy], [du_y/dx, du_y/dy]
    grad_u_x = torch.autograd.grad(
        u_pinn[:, 0], X_with_grad, 
        grad_outputs=torch.ones_like(u_pinn[:, 0]), 
        create_graph=True
    )[0]
    
    grad_u_y = torch.autograd.grad(
        u_pinn[:, 1], X_with_grad, 
        grad_outputs=torch.ones_like(u_pinn[:, 1]), 
        create_graph=True
    )[0]

    # Calcoliamo una loss fittizia che unisce:
    # A. La perdita fisica del gradiente spaziale della PINN (coerenza del continuo)
    loss_physics = torch.mean(grad_u_x**2 + grad_u_y**2)
    # B. La perdita di contorno o sui dati sullo spostamento finale corretto da MGN
    u_target = torch.zeros_like(u_final) # Es. vincolo di spostamento nullo
    loss_boundary = torch.mean((u_final - u_target)**2)
    
    loss_total = loss_physics + loss_boundary

    print(f"\n[LOSS CALCOLATA]: {loss_total.item():.6f}")
    print(f" - Loss Fisica (PINN continuous gradient): {loss_physics.item():.6f}")
    print(f" - Loss Contorno/Dati (on Hybrid displacement): {loss_boundary.item():.6f}")

    # 7. Backward Pass (Retropropagazione)
    optimizer = torch.optim.Adam(hybrid_model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss_total.backward()

    # Verifica dei gradienti
    pinn_has_grad = any(p.grad is not None for p in pinn_model.parameters())
    mgn_has_grad = any(p.grad is not None for p in mgn_model.parameters())

    print("\n" + "-" * 70)
    print("VERIFICA FLUSSO GRADIENTI (BACKPROPAGATION)")
    print("-" * 70)
    print(f"PINN Backbone ha gradienti:       {'PASS' if pinn_has_grad else 'FAIL'}")
    print(f"MeshGraphNet ha gradienti:        {'PASS' if mgn_has_grad else 'FAIL'}")
    print("=" * 80)
