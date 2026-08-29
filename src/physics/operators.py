import torch
from src.mechanics.deformation import compute_element_deformation_gradients
from src.utils import grad, voigt_tensor
from .equations import HUGO, FungEnergy_2D, Material

def haslach_constitutive_evolution_2D(
    u_total: torch.Tensor,          # Spostamento ibrido incognito al passo corrente (u_pinn + delta_u_gnn)
    X_ref: torch.Tensor,             # Coordinate materiali di riferimento dei nodi (X)
    E_prev_tensor: torch.Tensor,      # Tensore di Green-Lagrange Voigt salvato al passo t_{n-1} (congelato!)
    k_relax: float,                  # Modulo di rilassamento viscoelastico (k)
    c_isotropic: float,              # Contributo isotropo della matrice (c)
    k2_fiber: float,                 # Parametro esponenziale delle fibre (k2)
    beta_fiber: float,               # Angolo di orientamento delle fibre (beta) in radianti
    S_applied: torch.Tensor,   # Sforzo esterno applicato Voigt [S11, S22, S12] al passo corrente
    dt: float,                       # Passo temporale discreto (delta_t)
    reg_Hessian: float = 1e-2,       # Stabilizzazione diagonalizzata contro det_H = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    # Calcola il tensore di Green-Lagrange Voigt corrente E^n e il residuo di Haslach per l'evoluzione costitutiva viscoelastica 2D.
    num_elements = u_total.shape[0]
    
    # 1. Costruzione del tensore identità 3D per il calcolo di F e E
    I = torch.eye(3, device=u_total.device).unsqueeze(0).repeat(num_elements, 1, 1)

    # 2. Calcolo del Tensore di Deformazione F^n = I + grad(u_total, X_ref)
    F = I + grad(u_total, X_ref)  # [num_elementi, 3, 3]
    
    # 3. Calcolo del Tensore di Green-Lagrange corrente E^n = 0.5 * (F^T * F - I) (3D per incompressibilità)
    E_tensor = 0.5 * (torch.bmm(F.transpose(1, 2), F) - I)
    E_voigt = voigt_tensor(E_tensor, True)  # [num_elementi, 3]

    E_dot = (E_tensor - E_prev_tensor) / dt
    E_dot_voigt = voigt_tensor(E_dot, True)
    
    # 5. Calcolo dei parametri delle fibre anisotropiche HGO (Passo 515 del paper)
    c1 = 4.0 * k2_fiber * (torch.cos(beta_fiber) ** 4)
    c2 = 4.0 * k2_fiber * (torch.sin(beta_fiber) ** 4)
    c3 = 8.0 * k2_fiber * (torch.cos(beta_fiber) ** 2) * (torch.sin(beta_fiber) ** 2)
        
    # Sforzo esterno applicato (componenti principali Sh e Sz)
    S_applied_voigt = voigt_tensor(S_applied, True)

    # 6. Calcolo del target di deformazione (dot{E}_target)
    # Questo è il target di deformazione che il modello deve raggiungere
    E_dot_target = HUGO(FungEnergy_2D(c_isotropic, c1, c2, c3), k_relax).haslach_equation(E_voigt, S_applied_voigt)

    # Residuo di Haslach
    residual = E_dot_voigt - E_dot_target
    
    return E_dot_voigt, residual

OPERATOR_REGISTRY = {
    "viscoelastic_residual_Fung_2D": haslach_constitutive_evolution_2D,
}