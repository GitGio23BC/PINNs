##### FEM Dataset Generation

* **What**: The commit add a dataset generator that uses a FEM solver (Newton-Raphson loop) to get the displacement "u", its gradient "grad\_u", the external force and the energy computed in the analytical way.
* **When**: 03/10/2026
* **Who**: Giorgio
* **Where**: src/loader and src/equation
* **Why**: We need a ground truth to produce the data loss on the PMGN regularisation
* **How**: Using a Newton-Raphson loop, we take a mesh with the left side clamped (that will be Dirichlet condition) and the external force applied on the right side (that will be Newman condition). The Newton-Raphson loop solves the problem imposing the force balance (R=F\_ext-F\_in <= tollerance) for each time instant. The physical quantities used and how they are computed is described below:

  * *Force residue "R"* -> R = F\_ext - F\_in; it's the residue that drive the loop optimisation.
  * *External force "F\_ext"* -> F\_ext = element\_area \* (sigma\_max \* sin(πt/T)); it's the know outside force applied in a half a cicle period
  * *Internal energy "ψ"* -> Ogden energy computed using the paper analytical formula.
  * *Total internal energy "Π"* -> Π = sum(ψ\_e\*A\_e) is the total energy inside the deformed body computed considering the Odgen energy for each mesh element times the fixed area of each mesh element.
  * *Internal force "F\_in"* -> F\_int = dΠ/du; the internal force as reaction of the external force, it is computed using the work-force duality F = -dU/dx.
  * *Stiffness matrix "K"*: K = dF\_int/du = d^2Π/du^2, it's shows how much the body change it's displacement under a force (F=KΔu). In hyper-elastic model isn't constant and depend on the internal forces, displacement etc. To compute the general definition is applied knowing the work-force duality.
  * *Displacement variation "Δu"* -> Δu = K^-1 \* R; the displacement variation, it changes with time and the current condition. It's computed suing a linear approximation with the stiffness matrix inversion.
  * *Displacement "u" ->* u(k+1) = u(k) + αΔu; the current displacement incremented each step in a secure way to avoid NaN values.
  * *Displacement gradient "grad\_u"* -> grad\_u = u \* dN/dx; how much the displacement changes in the mesh, computed using a fixed gradient stencil get from the know geometry of each mesh element (triangular)
* Sources: https://learnfea.com/newton-raphson-method-for-nonlinear-fea/

