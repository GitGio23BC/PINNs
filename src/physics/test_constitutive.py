import numpy as np

from .neo_hookean import NeoHookean
from .mooney_rivlin import MooneyRivlin


def print_result(model, name, F):

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    J = np.linalg.det(F)

    print("\nDeformation gradient F:")
    print(F)

    print("\ndet(F):")
    print(J)

    try:

        energy = model.strain_energy(F)
        P = model.first_piola_kirchhoff(F)
        sigma = model.cauchy_stress(F)

        print("\nStrain energy Ψ:")
        print(energy)

        print("\nFirst Piola-Kirchhoff stress P:")
        print(P)

        print("\nCauchy stress σ:")
        print(sigma)

    except ValueError as error:

        print("\nERROR:")
        print(error)


def main():

    # ---------------------------------------------------------
    # TEST 1: undeformed configuration
    # ---------------------------------------------------------

    F_identity = np.eye(3)

    # ---------------------------------------------------------
    # TEST 2: simple uniaxial deformation
    # ---------------------------------------------------------

    F_stretch = np.array([
        [1.10, 0.00, 0.00],
        [0.00, 1.00, 0.00],
        [0.00, 0.00, 1.00]
    ])

    # ---------------------------------------------------------
    # TEST 3: general deformation
    # ---------------------------------------------------------

    F_general = np.array([
        [1.10, 0.05, 0.00],
        [0.00, 0.95, 0.02],
        [0.00, 0.00, 1.05]
    ])

    # ---------------------------------------------------------
    # TEST 4: invalid deformation
    # ---------------------------------------------------------

    F_invalid = np.array([
        [-1.00, 0.00, 0.00],
        [0.00, 1.00, 0.00],
        [0.00, 0.00, 1.00]
    ])

    # ---------------------------------------------------------
    # MATERIALS
    # ---------------------------------------------------------

    neo_hookean = NeoHookean(
        mu=1.0,
        bulk_modulus=100.0
    )

    mooney_rivlin = MooneyRivlin(
        C1=0.5,
        C2=0.5,
        bulk_modulus=100.0
    )

    models = [
        (neo_hookean, "Neo-Hookean"),
        (mooney_rivlin, "Mooney-Rivlin")
    ]

    tests = [
        (F_identity, "Identity"),
        (F_stretch, "Uniaxial stretch"),
        (F_general, "General deformation"),
        (F_invalid, "Invalid deformation")
    ]

    # ---------------------------------------------------------
    # RUN TESTS
    # ---------------------------------------------------------

    for F, test_name in tests:

        print("\n")
        print("#" * 70)
        print(f"TEST: {test_name}")
        print("#" * 70)

        for model, model_name in models:

            print_result(
                model,
                model_name,
                F
            )


if __name__ == "__main__":
    main()