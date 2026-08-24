import numpy as np

from .hgo import HGO


def fiber_invariants(model, F):
    """
    Compute the fiber invariants I4 for the
    two HGO fiber families.
    """

    C = F.T @ F

    I4_a = model.a0 @ C @ model.a0
    I4_b = model.b0 @ C @ model.b0

    return I4_a, I4_b


def fiber_strains(model, F):
    """
    Compute the Green-Lagrange strain associated
    with each fiber family.

        E_f = 0.5 * (I4 - 1)
    """

    I4_a, I4_b = fiber_invariants(model, F)

    E_a = 0.5 * (I4_a - 1.0)
    E_b = 0.5 * (I4_b - 1.0)

    return E_a, E_b


def print_case(model, name, F):
    """
    Print all relevant quantities for one deformation.
    """

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    I4_a, I4_b = fiber_invariants(model, F)
    E_a, E_b = fiber_strains(model, F)

    psi_total = model.strain_energy(F)
    P = model.first_piola_kirchhoff(F)

    print("\nDeformation gradient F:")
    print(F)

    print("\ndet(F):")
    print(np.linalg.det(F))

    print("\nI4 family A:")
    print(I4_a)

    print("\nI4 family B:")
    print(I4_b)

    print("\nFiber strain E_A:")
    print(E_a)

    print("\nFiber strain E_B:")
    print(E_b)

    print("\nPositive fiber strain <E_A>:")
    print(max(E_a, 0.0))

    print("\nPositive fiber strain <E_B>:")
    print(max(E_b, 0.0))

    print("\nTotal strain energy Ψ:")
    print(psi_total)

    print("\nFirst Piola-Kirchhoff stress P:")
    print(P)


def main():

    # ---------------------------------------------------------
    # HGO parameters
    # ---------------------------------------------------------

    model = HGO(
        mu=1.0,
        bulk_modulus=100.0,
        k1=1.0,
        k2=10.0,
        fiber_angle=np.deg2rad(30.0)
    )

    print("=" * 70)
    print("HGO FIBER BEHAVIOR TEST")
    print("=" * 70)

    print("\nFiber A:")
    print(model.a0)

    print("\nFiber B:")
    print(model.b0)

    # ---------------------------------------------------------
    # TEST 1
    # Stretch along x
    # Both fiber families have an x component.
    # ---------------------------------------------------------

    F_x_stretch = np.array([
        [1.20, 0.00, 0.00],
        [0.00, 1.00, 0.00],
        [0.00, 0.00, 1.00]
    ])

    print_case(
        model,
        "TEST 1 - X DIRECTION STRETCH",
        F_x_stretch
    )

    # ---------------------------------------------------------
    # TEST 2
    # Stretch directly along fiber A
    # ---------------------------------------------------------

    a = model.a0

    stretch = 1.20

    F_a_stretch = (
        np.eye(3)
        +
        (stretch - 1.0)
        * np.outer(a, a)
    )

    print_case(
        model,
        "TEST 2 - STRETCH ALONG FIBER A",
        F_a_stretch
    )

    # ---------------------------------------------------------
    # TEST 3
    # Compression along fiber A
    # ---------------------------------------------------------

    compression = 0.80

    F_a_compression = (
        np.eye(3)
        +
        (compression - 1.0)
        * np.outer(a, a)
    )

    print_case(
        model,
        "TEST 3 - COMPRESSION ALONG FIBER A",
        F_a_compression
    )


if __name__ == "__main__":
    main()