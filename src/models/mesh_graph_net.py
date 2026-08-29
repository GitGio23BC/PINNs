import torch
import torch.nn as nn
import numpy as np

# ==============================================================================
# 1. MULTI-LAYER PERCEPTRON (MLP)
# ==============================================================================
class MLP(nn.Module):
    """
    Multilayer Perceptron (MLP) standard con Layer Normalization residua.
    Utilizzato come blocco costruttivo fondamentale per l'Encoder, i blocchi di
    Message Passing del Processor e il Decoder.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, use_layer_norm: bool = True):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.layer_norm = nn.LayerNorm(output_dim) if use_layer_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(self.network(x))
# ==============================================================================
# 2. ENCODER
# ==============================================================================
class Encoder(nn.Module):
    """
    MeshGraphNet Encoder.
    Mappa separatamente le feature fisiche dei nodi e degli archi in uno spazio
    latente condiviso di dimensione fissa (latent_dim).
    """
    def __init__(self, node_input_dim: int, edge_input_dim: int, latent_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.node_encoder = MLP(input_dim=node_input_dim, hidden_dim=hidden_dim, output_dim=latent_dim)
        self.edge_encoder = MLP(input_dim=edge_input_dim, hidden_dim=hidden_dim, output_dim=latent_dim)

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent_nodes = self.node_encoder(node_features)
        latent_edges = self.edge_encoder(edge_features)
        return latent_nodes, latent_edges
# ==============================================================================
# 3. MESSAGE PASSING BLOCK
# ==============================================================================
class MessagePassingBlock(nn.Module):
    """
    Singolo blocco di Message Passing (GNN) ispirato a MeshGraphNet.
    Aggiorna le feature latenti degli archi e dei nodi tramite scambio locale:
    1. Edge update: e'_{ij} = e_{ij} + MLP_e( [v_i, v_j, e_{ij}] )
    2. Message aggregation: m_i = sum_{j in N(i)} e'_{ij}
    3. Node update: v'_i = v_i + MLP_v( [v_i, m_i] )
    """
    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        # Input dell'Edge MLP: nodi adiacenti (2 * latent_dim) + archi correnti (latent_dim)
        self.edge_mlp = MLP(input_dim=latent_dim * 3, hidden_dim=hidden_dim, output_dim=latent_dim)
        # Input del Node MLP: nodo corrente (latent_dim) + messaggi aggregati (latent_dim)
        self.node_mlp = MLP(input_dim=latent_dim * 2, hidden_dim=hidden_dim, output_dim=latent_dim)

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor, 
                senders: torch.Tensor, receivers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. Edge Update
        src_features = node_features[senders]
        dst_features = node_features[receivers]
        edge_input = torch.cat([src_features, dst_features, edge_features], dim=-1)
        updated_edges = edge_features + self.edge_mlp(edge_input) # Connessione residua

        # 2. Aggregazione dei messaggi (Somma)
        num_nodes = node_features.shape[0]
        aggregated_messages = torch.zeros(num_nodes, edge_features.shape[-1], device=node_features.device)
        aggregated_messages.index_add_(0, receivers, updated_edges)

        # 3. Node Update
        node_input = torch.cat([node_features, aggregated_messages], dim=-1)
        updated_nodes = node_features + self.node_mlp(node_input) # Connessione residua

        return updated_nodes, updated_edges
# ==============================================================================
# 4. PROCESSOR
# ==============================================================================
class MeshGraphNetProcessor(nn.Module):
    """
    MeshGraphNet Processor.
    Composto da L blocchi di Message Passing in serie per propagare le informazioni
    geometriche e fisiche attraverso la mesh a lungo raggio.
    """
    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64, num_layers: int = 15):
        super().__init__()
        self.layers = nn.ModuleList([
            MessagePassingBlock(latent_dim=latent_dim, hidden_dim=hidden_dim)
            for _ in range(num_layers)
        ])

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor, 
                senders: torch.Tensor, receivers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_nodes, h_edges = node_features, edge_features
        for layer in self.layers:
            h_nodes, h_edges = layer(h_nodes, h_edges, senders, receivers)
        return h_nodes, h_edges
# ==============================================================================
# 5. DECODER
# ==============================================================================
class Decoder(nn.Module):
    """
    MeshGraphNet Decoder.
    Converte lo stato latente dei nodi (dopo il processore) in quantità fisiche reali.
    Nel nostro caso, predice l'incremento di spostamento correttivo (du).
    """
    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64, output_dim: int = 2):
        super().__init__()
        # Il Decoder solitamente non usa Layer Normalization nell'ultimo strato
        self.mlp = MLP(input_dim=latent_dim, hidden_dim=hidden_dim, output_dim=output_dim, use_layer_norm=False)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        return self.mlp(node_features)
# ==============================================================================
# 6. PIPELINE IBRIDA: MESHGRAPHNET + PINN
# ==============================================================================
class MeshGraphNet(nn.Module):
    """
    Modello MeshGraphNet completo integrato con PINN.
    Prende in ingresso la mesh (grafo), le feature estratte e l'output della PINN,
    le mappa nello spazio latente, esegue il Message Passing e decodifica lo spostamento.
    """
    def __init__(self, node_input_dim: int, edge_input_dim: int, 
                 latent_dim: int = 32, hidden_dim: int = 64, 
                 num_processor_layers: int = 15, output_dim: int = 2):
        super().__init__()
        self.encoder = Encoder(node_input_dim=node_input_dim, edge_input_dim=edge_input_dim, 
                               latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.processor = MeshGraphNetProcessor(latent_dim=latent_dim, hidden_dim=hidden_dim, 
                                               num_layers=num_processor_layers)
        self.decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor, 
                senders: torch.Tensor, receivers: torch.Tensor) -> torch.Tensor:
        # Encode
        latent_nodes, latent_edges = self.encoder(node_features, edge_features)
        # Process (Message Passing x L)
        latent_nodes, latent_edges = self.processor(latent_nodes, latent_edges, senders, receivers)
        # Decode
        output = self.decoder(latent_nodes)
        return output
# ==============================================================================
# MOCK PINN CONTINUA
# ==============================================================================
class MockContinuousPINN(nn.Module):
    """
    Rete PINN continua giocattolo.
    Rappresenta la mappatura continua u_pinn = PINN(X_mesh) che rispetta le PDE.
    Genera spostamenti basandosi sulle coordinate materiali continue della mesh.
    """
    def __init__(self, input_dim: int = 2, output_dim: int = 2, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(), # Le PINN preferiscono Tanh o Sin per derivabilità liscia
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
# ==============================================================================
# FUNZIONE PER L'ESTRAZIONE DELLE FEATURE DEL GRAFO (ENCODING FISICO)
# ==============================================================================
def extract_graph_features(X_reference: torch.Tensor, x_current: torch.Tensor, 
                           u_pinn: torch.Tensor, senders: torch.Tensor, 
                           receivers: torch.Tensor, node_types: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Estrae le feature spaziali ed equivarianti per nodi e archi seguendo la fisica di MeshGraphNet.
    
    Feature Archi:
      - Spostamento relativo nello spazio materiale di riferimento: U_ij = X_i - X_j
      - Norma dello spostamento materiale: |U_ij|
      - Spostamento relativo nello spazio reale corrente: u_ij = x_i - x_j
      - Norma dello spostamento reale: |u_ij|
      Dimensione totale = 2 + 1 + 2 + 1 = 6 feature.
      
    Feature Nodi:
      - Tipo di nodo (one-hot o scalare, es: 0=normale, 1=contorno/cinematico)
      - Spostamento continuo calcolato dalla PINN: u_pinn = [u_x, u_y]
      Dimensione totale = 1 + 2 = 3 feature.
    """
    # 1. Feature degli archi (Edges)
    U_ij = X_reference[senders] - X_reference[receivers] # (E, 2)
    norm_U = torch.norm(U_ij, dim=-1, keepdim=True)       # (E, 1)

    u_ij = x_current[senders] - x_current[receivers]     # (E, 2)
    norm_u = torch.norm(u_ij, dim=-1, keepdim=True)       # (E, 1)

    edge_features = torch.cat([U_ij, norm_U, u_ij, norm_u], dim=-1) # (E, 6)

    # 2. Feature dei nodi (Nodes)
    # Concateniamo il tipo di nodo e lo spostamento continuo predetto dalla PINN
    node_features = torch.cat([node_types, u_pinn], dim=-1) # (N, 3)

    return node_features, edge_features
# ==============================================================================
# TEST & VALIDAZIONE COMPLETA DELLA PIPELINE
# ==============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("           COMPILING & TESTING COMPLETE HYBRID PINN + MESHGRAPHNET")
    print("=" * 80)

    # Impostiamo seed per riproducibilità
    torch.manual_seed(42)
    np.random.seed(42)

    # --------------------------------------------------------------------------
    # 1. Configurazione della Mesh giocattolo (4 nodi, 6 archi orientati)
    # --------------------------------------------------------------------------
    num_nodes = 4
    # Coordinate materiali di riferimento (X)
    X_reference = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 1.0],
        [1.5, 1.0]
    ], dtype=torch.float32, requires_grad=True) # Abilitiamo i gradienti per la sensibilità spaziale

    # Spostamento iniziale nullo (quindi x_corrente = X_riferimento inizialmente)
    x_current = X_reference.clone().detach().requires_grad_(True)

    # Tipi di nodi (es: nodo 0,1 fissati/cinematici (tipo 1.0), nodi 2,3 liberi di deformarsi (tipo 0.0))
    node_types = torch.tensor([[1.0], [1.0], [0.0], [0.0]], dtype=torch.float32)

    # Definizione topologica della Mesh: archi orientati (senders -> receivers)
    senders = torch.tensor([0, 1, 1, 2, 2, 3], dtype=torch.long)
    receivers = torch.tensor([1, 0, 2, 1, 3, 2], dtype=torch.long)

    print(f"Mesh configurata con {num_nodes} nodi e {len(senders)} archi orientati.")

    # --------------------------------------------------------------------------
    # 2. Inizializzazione della PINN Continua e di MeshGraphNet
    # --------------------------------------------------------------------------
    pinn_model = MockContinuousPINN(input_dim=2, output_dim=2, hidden_dim=16)
    
    # Feature fisiche dedotte da extract_graph_features
    node_feature_dim = 3  # tipo_nodo (1) + u_pinn (2)
    edge_feature_dim = 6  # U_ij (2) + |U_ij| (1) + u_ij (2) + |u_ij| (1)
    latent_dim = 32
    hidden_dim = 64
    num_processor_layers = 4 # usiamo 4 layer per un test rapido e leggero

    mgn_model = MeshGraphNet(
        node_input_dim=node_feature_dim,
        edge_input_dim=edge_feature_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_processor_layers=num_processor_layers,
        output_dim=2 # L'output decodificato è l'incremento di spostamento correttivo [du_x, du_y]
    )

    print("\nModelli inizializzati correttamente:")
    print(f" - PINN continua (input: 2, output: 2)")
    print(f" - MeshGraphNet (node_input_dim: {node_feature_dim}, edge_input_dim: {edge_feature_dim}, processor_layers: {num_processor_layers})")

    # --------------------------------------------------------------------------
    # 3. Forward Pass: Flusso Ibrido PINN + GNN
    # --------------------------------------------------------------------------
    # Fase A: Otteniamo lo stato continuo della PINN interrogandolo sui nodi materiali X
    u_pinn = pinn_model(X_reference)
    print(f"\nDislocamento continuo generato dalla PINN sui nodi materiali X:")
    print(u_pinn)

    # Fase B: Estrazione delle feature strutturate per il grafo
    node_features, edge_features = extract_graph_features(
        X_reference=X_reference,
        x_current=x_current,
        u_pinn=u_pinn,
        senders=senders,
        receivers=receivers,
        node_types=node_types
    )
    print(f"\nFeature del grafo estratte:")
    print(f" - Node features shape: {node_features.shape}")
    print(f" - Edge features shape: {edge_features.shape}")

    # Fase C: Esecuzione di MeshGraphNet per predire lo spostamento correttivo final
    mgn_output = mgn_model(node_features, edge_features, senders, receivers)
    print(f"\nSpostamento correttivo decodificato da MeshGraphNet (Output):")
    print(mgn_output)

    # --------------------------------------------------------------------------
    # 4. Verifica dei Gradienti (Backpropagation Test)
    # --------------------------------------------------------------------------
    # Calcoliamo una loss fittizia che dipende dall'output combinato e dalla fisica
    # Ad esempio, minimizziamo la norma L2 dello spostamento combinato (o un'energia elastica semplice)
    target_displacement = torch.tensor([[0.1, -0.1], [0.2, 0.0], [-0.1, 0.3], [0.0, 0.0]])
    loss = nn.MSELoss()(u_pinn + mgn_output, target_displacement)

    print(f"\nLoss calcolata: {loss.item():.6f}")

    # Eseguiamo il backward pass
    loss.backward()

    # Controlliamo che i gradienti si propaghino correttamente a ritroso sia in MGN che nella PINN
    pinn_grad_ok = pinn_model.net[0].weight.grad is not None
    mgn_encoder_grad_ok = mgn_model.encoder.node_encoder.network[0].weight.grad is not None
    mgn_processor_grad_ok = mgn_model.processor.layers[0].edge_mlp.network[0].weight.grad is not None
    mgn_decoder_grad_ok = mgn_model.decoder.mlp.network[0].weight.grad is not None

    print("\n" + "-" * 70)
    print("VERIFICA FLUSSO GRADIENTI (BACKPROPAGATION)")
    print("-" * 70)
    print(f"PINN gradiente propagato:          " + ("PASS" if pinn_grad_ok else "FAIL"))
    print(f"MGN Encoder gradiente propagato:  " + ("PASS" if mgn_encoder_grad_ok else "FAIL"))
    print(f"MGN Processor gradiente propagato:" + ("PASS" if mgn_processor_grad_ok else "FAIL"))
    print(f"MGN Decoder gradiente propagato:  " + ("PASS" if mgn_decoder_grad_ok else "FAIL"))

    # Verifica finale del test
    all_passed = pinn_grad_ok and mgn_encoder_grad_ok and mgn_processor_grad_ok and mgn_decoder_grad_ok
    print("\n" + "=" * 70)
    if all_passed:
        print("RISULTATO FINALE DEL TEST: PASS")
    else:
        print("RISULTATO FINALE DEL TEST: FAIL")
    print("=" * 70)
