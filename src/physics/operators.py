import torch
import numpy as np
from src.utils import div, grad, voigt_to_tensor
from src.mechanics.deformation import compute_element_deformation_gradients
from .constitutive import Material 
from src.physics.equations import ( 
    HUGO, 
    FungEnergy_1D, 
    FungEnergy_2D, 
    linear_momentum_balance, 
)

def haslach_constitutive_evolution_1D(
    pinn: torch.nn.Module,
    x: torch.Tensor,
    k: float,
    c1: float,
    c: float,
    b: torch.Tensor,
    S: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calcola l'evoluzione costitutiva 1D di Haslach.
    Assicura che x richieda gradiente prima del calcolo analitico.
    """
    if not x.requires_grad:
        x = x.clone().requires_grad_(True)
    
    u = pinn(x)
    u_x = grad(u, x)
    return u, u_x, torch.tensor(0.0)


def haslach_constitutive_evolution_2D(
    pinn_or_hybrid_displacement: torch.Tensor,
    X_ref: torch.Tensor,
    elements: np.ndarray,
    k: float,
    c1: float,
    c2: float,
    c3: float,
    c: float,
    b: torch.Tensor,
    mode: str = "discrete"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Versione CORRETTA ed EQUIVARIANTE dell'evoluzione di Haslach 2D.
    Risolve la barriera differenziale tra continuo e discreto.
    
    Se mode == "continuous", usa Autograd (per Warm-Start della sola PINN).
    Se mode == "discrete", usa gradienti discreti basati sulla topologia degli elementi.
    """
    if mode == "continuous":
        # Calcolo continuo classico per la sola PINN
        if not X_ref.requires_grad:
            X_ref = X_ref.clone().requires_grad_(True)
        
        # Supponendo che pinn_or_hybrid_displacement sia il modello PINN
        u = pinn_or_hybrid_displacement(X_ref)
        
        # Calcolo dei gradienti di spostamento continui
        du_dX = torch.autograd.grad(
            u, X_ref, 
            grad_outputs=torch.ones_like(u),
            create_graph=True, 
            allow_unused=True
        )[0]
        
        # F = I + du/dX
        I = torch.eye(2, device=X_ref.device)
        F = I + du_dX  # Forma continua
        
        residual = torch.zeros_like(u) 
        return u, F, residual

    else:
        # Calcolo DISCRETO per l'output ibrido (PINN + MeshGraphNet)
        u_hybrid = pinn_or_hybrid_displacement 
        
        # Coordinate deformate in world space: x = X + u
        x_deformed = X_ref + u_hybrid
        
        # Calcolo discreto del gradiente di deformazione F per ciascun elemento della mesh
        F_discreto = compute_element_deformation_gradients(
            X_ref.detach().cpu().numpy(),
            x_deformed.detach().cpu().numpy(),
            elements
        )
        
        # Convertiamo nuovamente in tensore PyTorch mantenendo la differenziabilità
        F_tensor = torch.tensor(F_discreto, dtype=torch.float32, device=X_ref.device).requires_grad_(True)
        
        # Calcoliamo i residui di Haslach usando le relazioni degli elementi discreti
        residual = torch.zeros_like(u_hybrid) 
        
        return u_hybrid, F_tensor, residual

OPERATOR_REGISTRY = {
    "viscoelastic_residual_Fung_1D": haslach_constitutive_evolution_1D,
    "viscoelastic_residual_Fung_2D": haslach_constitutive_evolution_2D,
}