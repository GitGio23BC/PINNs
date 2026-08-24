import numpy as np


def deformation_gradient_2d(
    X_element: np.ndarray,
    x_element: np.ndarray
) -> np.ndarray:
    """
    Compute the 3D deformation gradient F for a 2D triangular element.

    Parameters
    ----------
    X_element : ndarray, shape (3, 2)
        Reference coordinates of the triangle nodes.

    x_element : ndarray, shape (3, 2)
        Deformed coordinates of the triangle nodes.

    Returns
    -------
    F : ndarray, shape (3, 3)
        Deformation gradient.

        The third direction is initially assumed undeformed:
            F[2,2] = 1

    Notes
    -----
    For a linear triangular element, the deformation gradient
    is constant inside the element.
    """

    X_element = np.asarray(X_element, dtype=float)
    x_element = np.asarray(x_element, dtype=float)

    if X_element.shape != (3, 2):
        raise ValueError(
            "X_element must have shape (3, 2)."
        )

    if x_element.shape != (3, 2):
        raise ValueError(
            "x_element must have shape (3, 2)."
        )

    # ---------------------------------------------------------
    # Reference edge matrix
    #
    # Each column represents one edge starting from node 0.
    #
    #        X2
    #        ●
    #       / \
    #      /   \
    #     /     \
    #    ●-------●
    #   X0       X1
    # ---------------------------------------------------------

    D_X = np.column_stack([
        X_element[1] - X_element[0],
        X_element[2] - X_element[0]
    ])

    # ---------------------------------------------------------
    # Deformed edge matrix
    # ---------------------------------------------------------

    D_x = np.column_stack([
        x_element[1] - x_element[0],
        x_element[2] - x_element[0]
    ])

    # ---------------------------------------------------------
    # Check that the reference element is not degenerate
    # ---------------------------------------------------------

    det_D_X = np.linalg.det(D_X)

    if abs(det_D_X) < 1e-12:
        raise ValueError(
            "Degenerate reference triangle: "
            "area is approximately zero."
        )

    # ---------------------------------------------------------
    # 2D deformation gradient
    #
    #        dx = F dX
    #
    # Therefore:
    #
    #        F = D_x @ inv(D_X)
    # ---------------------------------------------------------

    F_2d = D_x @ np.linalg.inv(D_X)

    # ---------------------------------------------------------
    # Embed the 2D deformation gradient into 3D
    # ---------------------------------------------------------

    F = np.eye(3)

    F[:2, :2] = F_2d

    return F


def compute_element_deformation_gradients(
    X: np.ndarray,
    x: np.ndarray,
    elements: np.ndarray
) -> np.ndarray:
    """
    Compute the deformation gradient for every triangular element.

    Parameters
    ----------
    X : ndarray, shape (N, 2)
        Reference mesh nodes.

    x : ndarray, shape (N, 2)
        Deformed mesh nodes.

    elements : ndarray, shape (M, 3)
        Triangle connectivity.

    Returns
    -------
    F_all : ndarray, shape (M, 3, 3)
        Deformation gradient for each element.
    """

    X = np.asarray(X, dtype=float)
    x = np.asarray(x, dtype=float)
    elements = np.asarray(elements, dtype=int)

    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(
            "X must have shape (N, 2)."
        )

    if x.shape != X.shape:
        raise ValueError(
            "x must have the same shape as X."
        )

    if elements.ndim != 2 or elements.shape[1] != 3:
        raise ValueError(
            "elements must have shape (M, 3)."
        )

    F_all = []

    for element in elements:

        node_ids = element

        X_element = X[node_ids]
        x_element = x[node_ids]

        F = deformation_gradient_2d(
            X_element,
            x_element
        )

        F_all.append(F)

    return np.asarray(F_all)


def displacement_field(
    X: np.ndarray,
    x: np.ndarray
) -> np.ndarray:
    """
    Compute nodal displacement field.

        u = x - X
    """

    X = np.asarray(X, dtype=float)
    x = np.asarray(x, dtype=float)

    if X.shape != x.shape:
        raise ValueError(
            "X and x must have the same shape."
        )

    return x - X


if __name__ == "__main__":

    print("=" * 70)
    print("DEFORMATION MODULE TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Simple reference triangle
    # ---------------------------------------------------------

    X_element = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ])

    # Apply a known deformation:
    #
    # x = F X
    #
    # with:
    #
    # F = [[1.2, 0.1],
    #      [0.0, 0.9]]
    #

    F_expected = np.array([
        [1.2, 0.1],
        [0.0, 0.9]
    ])

    x_element = (F_expected @ X_element.T).T

    # ---------------------------------------------------------
    # Compute deformation gradient
    # ---------------------------------------------------------

    F = deformation_gradient_2d(
        X_element,
        x_element
    )

    print("\nReference coordinates X:")
    print(X_element)

    print("\nDeformed coordinates x:")
    print(x_element)

    print("\nExpected deformation gradient:")
    print(F_expected)

    print("\nComputed deformation gradient F:")
    print(F)

    # ---------------------------------------------------------
    # Error
    # ---------------------------------------------------------

    expected_3d = np.eye(3)
    expected_3d[:2, :2] = F_expected

    error = np.max(np.abs(F - expected_3d))

    print("\nMaximum absolute error:")
    print(error)

    if error < 1e-12:
        print("\nRESULT: PASS")
    else:
        print("\nRESULT: FAIL")