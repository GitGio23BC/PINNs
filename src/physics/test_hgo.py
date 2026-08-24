import numpy as np

from .hgo import HGO


def main():

    hgo = HGO(
        mu=1.0,
        bulk_modulus=100.0,
        k1=1.0,
        k2=10.0,
        fiber_angle=np.deg2rad(30.0)
    )

    print("=" * 70)
    print("HGO MODEL TEST")
    print("=" * 70)

    print("\nFiber direction a0:")
    print(hgo.a0)

    print("\nFiber direction b0:")
    print(hgo.b0)

    # ---------------------------------------------------------
    # Identity
    # ---------------------------------------------------------

    F_identity = np.eye(3)

    print("\n" + "-" * 70)
    print("IDENTITY")
    print("-" * 70)

    print("\ndet(F):")
    print(np.linalg.det(F_identity))

    print("\nΨ:")
    print(hgo.strain_energy(F_identity))

    print("\nP:")
    print(hgo.first_piola_kirchhoff(F_identity))

    # ---------------------------------------------------------
    # Stretch along x
    # ---------------------------------------------------------

    F_stretch = np.array([
        [1.10, 0.00, 0.00],
        [0.00, 1.00, 0.00],
        [0.00, 0.00, 1.00]
    ])

    print("\n" + "-" * 70)
    print("UNIAXIAL STRETCH")
    print("-" * 70)

    print("\nΨ:")
    print(hgo.strain_energy(F_stretch))

    print("\nP:")
    print(hgo.first_piola_kirchhoff(F_stretch))


if __name__ == "__main__":
    main()