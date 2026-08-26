import numpy as np


def compute_element_stress_gradient(
    X_element: np.ndarray,
    P_element: np.ndarray
) -> np.ndarray:
    """
    Compute the spatial gradient of a linearly interpolated
    first Piola-Kirchhoff stress field over a 2D triangular element.

    Parameters
    ----------
    X_element : ndarray, shape (3, 2)
        Reference coordinates of the triangle nodes.

    P_element : ndarray, shape (3, 3, 3)
        First Piola-Kirchhoff stress at the three element nodes.

        P_element[a, i, j] =
            P_ij at node a.

    Returns
    -------
    dP_dX : ndarray, shape (3, 3, 2)

        dP_dX[i, j, k] = ∂P_ij / ∂X_k

        where:
            k = 0 -> derivative with respect to X
            k = 1 -> derivative with respect to Y

    Notes
    -----
    Because linear triangular shape functions have constant
    gradients, the stress gradient is constant inside the element.
    """

    X_element = np.asarray(X_element, dtype=float)
    P_element = np.asarray(P_element, dtype=float)

    if X_element.shape != (3, 2):
        raise ValueError(
            "X_element must have shape (3, 2)."
        )

    if P_element.shape != (3, 3, 3):
        raise ValueError(
            "P_element must have shape (3, 3, 3)."
        )

    # ---------------------------------------------------------
    # Reference edge matrix
    #
    # D_X =
    #
    # [ X1-X0   X2-X0 ]
    # [ Y1-Y0   Y2-Y0 ]
    #
    # ---------------------------------------------------------

    D_X = np.column_stack([
        X_element[1] - X_element[0],
        X_element[2] - X_element[0]
    ])

    det_D_X = np.linalg.det(D_X)

    if abs(det_D_X) < 1e-12:
        raise ValueError(
            "Degenerate reference triangle."
        )

    # ---------------------------------------------------------
    # Gradients of linear shape functions
    #
    # N1, N2, N3 are linear over the triangle.
    #
    # grad(N_a) = [dN_a/dX, dN_a/dY]
    #
    # ---------------------------------------------------------

    grad_N = np.zeros((3, 2))

    # N0 = 1 - xi - eta
    # N1 = xi
    # N2 = eta

    inv_D_X = np.linalg.inv(D_X)

    grad_N[1] = inv_D_X[0]
    grad_N[2] = inv_D_X[1]
    grad_N[0] = -grad_N[1] - grad_N[2]

    # ---------------------------------------------------------
    # Stress gradient
    #
    # P(X) = sum_a N_a(X) P_a
    #
    # therefore
    #
    # dP/dX_k = sum_a dN_a/dX_k P_a
    #
    # ---------------------------------------------------------

    dP_dX = np.zeros((3, 3, 2))

    for a in range(3):

        for k in range(2):

            dP_dX[:, :, k] += (
                grad_N[a, k] * P_element[a]
            )

    return dP_dX


def compute_divergence_2d(
    dP_dX: np.ndarray
) -> np.ndarray:
    """
    Compute the divergence of the first Piola-Kirchhoff
    stress tensor in a 2D reference configuration.

    The equilibrium equation is:

        div(P) + b = 0

    For a 3D stress tensor varying with X and Y:

        div(P)_i =
            ∂P_i1/∂X + ∂P_i2/∂Y

    Parameters
    ----------
    dP_dX : ndarray, shape (3, 3, 2)

        dP_dX[i, j, k] =
            ∂P_ij / ∂X_k

    Returns
    -------
    divergence : ndarray, shape (3,)

        Divergence of P.
    """

    dP_dX = np.asarray(dP_dX, dtype=float)

    if dP_dX.shape != (3, 3, 2):
        raise ValueError(
            "dP_dX must have shape (3, 3, 2)."
        )

    divergence = np.zeros(3)

    # div(P)_i =
    #
    # ∂P_i1/∂X + ∂P_i2/∂Y
    #
    divergence = (
        dP_dX[:, 0, 0]
        +
        dP_dX[:, 1, 1]
    )

    return divergence


def compute_equilibrium_residual(
    dP_dX: np.ndarray,
    body_force: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the equilibrium residual.

    The strong-form equilibrium equation is:

        div(P) + b = 0

    Therefore:

        r_eq = div(P) + b

    Parameters
    ----------
    dP_dX : ndarray, shape (3, 3, 2)
        Gradient of the first Piola-Kirchhoff stress.

    body_force : ndarray, shape (3,), optional
        Body force per unit reference volume.

        If omitted, body_force = 0.

    Returns
    -------
    residual : ndarray, shape (3,)
        Equilibrium residual.
    """

    divergence = compute_divergence_2d(dP_dX)

    if body_force is None:
        body_force = np.zeros(3)

    body_force = np.asarray(body_force, dtype=float)

    if body_force.shape != (3,):
        raise ValueError(
            "body_force must have shape (3,)."
        )

    return divergence + body_force


def equilibrium_loss(
    residual: np.ndarray
) -> float:
    """
    Compute the mean squared equilibrium residual.
    """

    residual = np.asarray(residual, dtype=float)

    return float(
        np.mean(residual ** 2)
    )


# =============================================================
# TESTS
# =============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("EQUILIBRIUM MODULE TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Reference triangle
    # ---------------------------------------------------------

    X = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ])

    print("\nReference coordinates X:")
    print(X)

    # =========================================================
    # TEST 1
    #
    # Constant stress field
    #
    # P = constant
    #
    # Therefore:
    #
    # div(P) = 0
    #
    # =========================================================

    print("\n" + "-" * 70)
    print("TEST 1 - CONSTANT STRESS")
    print("-" * 70)

    P_constant = np.array([
        [
            [2.0, 0.5, 0.0],
            [0.5, 3.0, 0.0],
            [0.0, 0.0, 1.0]
        ],
        [
            [2.0, 0.5, 0.0],
            [0.5, 3.0, 0.0],
            [0.0, 0.0, 1.0]
        ],
        [
            [2.0, 0.5, 0.0],
            [0.5, 3.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
    ])

    dP_dX = compute_element_stress_gradient(
        X,
        P_constant
    )

    residual = compute_equilibrium_residual(
        dP_dX
    )

    loss = equilibrium_loss(residual)

    print("\nStress gradient:")
    print(dP_dX)

    print("\nEquilibrium residual:")
    print(residual)

    print("\nEquilibrium loss:")
    print(loss)

    expected = np.zeros(3)

    error = np.max(
        np.abs(residual - expected)
    )

    print("\nMaximum absolute error:")
    print(error)

    if error < 1e-12:
        print("\nRESULT: PASS")
    else:
        print("\nRESULT: FAIL")

    # =========================================================
    # TEST 2
    #
    # Variable stress:
    #
    # P11 = X
    # P22 = Y
    #
    # Therefore:
    #
    # div(P) = [1, 1, 0]
    #
    # =========================================================

    print("\n" + "-" * 70)
    print("TEST 2 - VARIABLE STRESS")
    print("-" * 70)

    P_variable = np.zeros((3, 3, 3))

    for a, (X_a, Y_a) in enumerate(X):

        P_variable[a, 0, 0] = X_a
        P_variable[a, 1, 1] = Y_a

    dP_dX = compute_element_stress_gradient(
        X,
        P_variable
    )

    residual = compute_equilibrium_residual(
        dP_dX
    )

    print("\nNodal stress values:")
    print(P_variable)

    print("\nStress gradient:")
    print(dP_dX)

    print("\nEquilibrium residual:")
    print(residual)

    expected = np.array([
        1.0,
        1.0,
        0.0
    ])

    error = np.max(
        np.abs(residual - expected)
    )

    print("\nExpected residual:")
    print(expected)

    print("\nMaximum absolute error:")
    print(error)

    if error < 1e-12:
        print("\nRESULT: PASS")
    else:
        print("\nRESULT: FAIL")

    # =========================================================
    # TEST 3
    #
    # Variable stress + body force
    #
    # div(P) = [1, 1, 0]
    #
    # b = [-1, -1, 0]
    #
    # Therefore:
    #
    # r_eq = div(P) + b = 0
    #
    # =========================================================

    print("\n" + "-" * 70)
    print("TEST 3 - VARIABLE STRESS + BODY FORCE")
    print("-" * 70)

    body_force = np.array([
        -1.0,
        -1.0,
        0.0
    ])

    residual = compute_equilibrium_residual(
        dP_dX,
        body_force
    )

    loss = equilibrium_loss(residual)

    print("\nBody force:")
    print(body_force)

    print("\nEquilibrium residual:")
    print(residual)

    print("\nEquilibrium loss:")
    print(loss)

    expected = np.zeros(3)

    error = np.max(
        np.abs(residual - expected)
    )

    print("\nExpected residual:")
    print(expected)

    print("\nMaximum absolute error:")
    print(error)

    if error < 1e-12:
        print("\nRESULT: PASS")
    else:
        print("\nRESULT: FAIL")