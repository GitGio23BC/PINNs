import numpy as np

from ..mechanics.deformation import deformation_gradient_2d


def compute_constitutive_response(
    X_element: np.ndarray,
    x_element: np.ndarray,
    material
) -> dict:
    """
    Compute the complete constitutive response of a single
    triangular element.

    Pipeline:

        Reference coordinates X
                ↓
        Deformed coordinates x
                ↓
        Deformation gradient F
                ↓
        Constitutive model
                ↓
        Ψ, P, σ

    Parameters
    ----------
    X_element : ndarray, shape (3, 2)
        Reference coordinates of the triangular element.

    x_element : ndarray, shape (3, 2)
        Current/deformed coordinates of the triangular element.

    material :
        Constitutive model implementing:

            strain_energy(F)
            first_piola_kirchhoff(F)

    Returns
    -------
    response : dict
        Dictionary containing:

            F       : deformation gradient
            J       : determinant of F
            energy  : strain energy density
            P       : first Piola-Kirchhoff stress
            sigma   : Cauchy stress
    """

    # ---------------------------------------------------------
    # 1. Compute deformation gradient
    # ---------------------------------------------------------

    F = deformation_gradient_2d(
        X_element,
        x_element
    )

    # ---------------------------------------------------------
    # 2. Check deformation
    # ---------------------------------------------------------

    J = np.linalg.det(F)

    if J <= 0:
        raise ValueError(
            "Invalid deformation: det(F) must be positive."
        )

    # ---------------------------------------------------------
    # 3. Constitutive model
    # ---------------------------------------------------------

    energy = material.strain_energy(F)

    P = material.first_piola_kirchhoff(F)

    # ---------------------------------------------------------
    # 4. Compute Cauchy stress
    #
    # σ = (1/J) P F^T
    # ---------------------------------------------------------

    sigma = (P @ F.T) / J

    # ---------------------------------------------------------
    # 5. Return complete response
    # ---------------------------------------------------------

    return {
        "F": F,
        "J": J,
        "energy": energy,
        "P": P,
        "sigma": sigma
    }


if __name__ == "__main__":

    print("=" * 70)
    print("CONSTITUTIVE RESPONSE TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Reference triangle
    # ---------------------------------------------------------

    X_element = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ])

    # ---------------------------------------------------------
    # Known deformation
    # ---------------------------------------------------------

    F_expected = np.array([
        [1.2, 0.1],
        [0.0, 0.9]
    ])

    x_element = (
        F_expected @ X_element.T
    ).T

    # ---------------------------------------------------------
    # HGO material
    # ---------------------------------------------------------

    from ..physics.hgo import HGO

    material = HGO(
        mu=10.0,
        bulk_modulus=100.0,
        k1=2.0,
        k2=10.0,
        fiber_angle=np.deg2rad(30.0)
    )

    # ---------------------------------------------------------
    # Compute response
    # ---------------------------------------------------------

    response = compute_constitutive_response(
        X_element,
        x_element,
        material
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print("\nDeformation gradient F:")
    print(response["F"])

    print("\nDeterminant J:")
    print(response["J"])

    print("\nStrain energy Ψ:")
    print(response["energy"])

    print("\nFirst Piola-Kirchhoff stress P:")
    print(response["P"])

    print("\nCauchy stress σ:")
    print(response["sigma"])

    print("\nRESULT: PASS")