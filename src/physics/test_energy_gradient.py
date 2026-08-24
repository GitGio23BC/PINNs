import numpy as np

from .neo_hookean import NeoHookean
from .mooney_rivlin import MooneyRivlin
from .hgo import HGO

def numerical_gradient(model, F, epsilon=1e-6):
    """
    Compute dΨ/dF numerically using central finite differences.
    """

    gradient = np.zeros_like(F)

    for i in range(F.shape[0]):
        for j in range(F.shape[1]):

            F_plus = F.copy()
            F_minus = F.copy()

            F_plus[i, j] += epsilon
            F_minus[i, j] -= epsilon

            psi_plus = model.strain_energy(F_plus)
            psi_minus = model.strain_energy(F_minus)

            gradient[i, j] = (
                psi_plus - psi_minus
            ) / (2.0 * epsilon)

    return gradient


def compare_model(model, name, F):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    analytical = model.first_piola_kirchhoff(F)

    numerical = numerical_gradient(model, F)

    difference = numerical - analytical

    absolute_error = np.max(
        np.abs(difference)
    )

    relative_error = (
        np.linalg.norm(difference)
        /
        max(np.linalg.norm(analytical), 1e-12)
    )

    print("\nAnalytical P:")
    print(analytical)

    print("\nNumerical dΨ/dF:")
    print(numerical)

    print("\nDifference:")
    print(difference)

    print("\nMaximum absolute error:")
    print(absolute_error)

    print("\nRelative error:")
    print(relative_error)

    if relative_error < 1e-5:
        print("\nRESULT: PASS")
    else:
        print("\nRESULT: FAIL")


def main():

    # A generic deformation gradient
    F = np.array([
        [1.10, 0.05, 0.00],
        [0.00, 0.95, 0.02],
        [0.00, 0.00, 1.05]
    ])

    hgo = HGO(
    mu=1.0,
    bulk_modulus=100.0,
    k1=1.0,
    k2=10.0,
    fiber_angle=np.deg2rad(30.0)
    )

    neo_hookean = NeoHookean(
        mu=1.0,
        bulk_modulus=100.0
    )

    mooney_rivlin = MooneyRivlin(
        C1=0.5,
        C2=0.5,
        bulk_modulus=100.0
    )

    compare_model(
        neo_hookean,
        "Neo-Hookean",
        F
    )

    compare_model(
        mooney_rivlin,
        "Mooney-Rivlin",
        F
    )

    compare_model(
        hgo,
        "HGO",
        F
    )

if __name__ == "__main__":
    main()