import torch
import torch.nn as nn

class ViscoelasticResidualEq33(nn.Module):
    """
    Calcola il residuo dell'Equazione 33 (Haslach et al.) usando una funzione 
    di energia Psi calcolata esternamente (es. Fung, HGO, Neo-Hooke, o Neural Network).
    """

    def __init__(self, k_rate: float):
        super().__init__()
        # k_rate: costante cinetica k dell'Equazione 33
        self.k_rate = torch.tensor(k_rate, dtype=torch.float32)

    def forward(self, E: torch.Tensor, E_dot: torch.Tensor, S_ext: torch.Tensor, Psi: torch.Tensor) -> torch.Tensor:
        """
        Input:
            E: Tensore di deformazione di Green-Lagrange corrente [3, 3] (requires_grad=True)
            E_dot: Tensore dE/dt corrente [3, 3]
            S_ext: Tensore degli sforzi esterni applicati S(t) [3, 3]
            Psi: Valore scalare dell'energia di deformazione calcolato ESTERNAMENTE
            
        Output:
            residual: Tensore del residuo dell'Eq. 33 [3, 3]. Se il sistema rispetta la dinamica, vale 0.
        """
        # 1. Calcolo della prima derivata dPsi/dE (tensione interna di richiamo)
        dPsi_dE = torch.autograd.grad(Psi, E, create_graph=True)[0]

        # 2. Calcolo della seconda derivata d2Psi/dE2 (Hessiano [9, 9])
        dPsi_dE_flat = dPsi_dE.reshape(-1)
        num_el = dPsi_dE_flat.numel()
        hessian = torch.zeros(num_el, num_el, device=E.device, dtype=E.dtype)

        for i in range(num_el):
            grad_i = torch.autograd.grad(dPsi_dE_flat[i], E, retain_graph=True)[0]
            hessian[i] = grad_i.reshape(-1)

        # 3. Termine inversione dell'Hessiano al quadrato: H^(-1) @ H^(-1)
        H_inv = torch.linalg.pinv(hessian)
        H_inv_squared = H_inv @ H_inv

        # 4. Forzante fuori equilibrio: [-S(t) + dPsi/dE]
        force_term = -S_ext + dPsi_dE
        force_term_flat = force_term.reshape(-1, 1)

        # 5. Termine destro dell'Equazione 33: RHS = -k * (H^-2) * [-S(t) + dPsi/dE]
        rhs_flat = -self.k_rate * (H_inv_squared @ force_term_flat)
        rhs = rhs_flat.reshape(3, 3)

        # 6. RESIDUO: dE/dt - RHS = 0
        residual = E_dot - rhs

        return residual