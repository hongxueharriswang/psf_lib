"""Demonstrate real-time state tracking with the particle filter."""
import numpy as np
from psf import (
    PSFModel, simulate_cohort, baum_welch, particle_filter,
)
from psf.states import JOINT_STATE_LABELS


def main():
    true_model = PSFModel(n_features=6)
    data = simulate_cohort(true_model, n_students=50, max_time=36)

    model = baum_welch(
        PSFModel(n_features=6),
        data["observations"],
        data["masks"],
        n_iter=10,
    )

    student_obs = data["observations"][0]
    student_mask = data["masks"][0]

    out = particle_filter(model, student_obs, student_mask, n_particles=1000)
    filtered = out["filtered_probs"]
    ess = out["ess"]

    print("Filtered state probabilities (last 5 timesteps):")
    for t in range(-5, 0):
        row = "  ".join(f"{v:.3f}" for v in filtered[t])
        print(f"  t={len(filtered)+t:3d}  {row}")

    print(f"\nFinal ESS: {ess[-1]:.1f} / 1000")
    print("\nState labels:", JOINT_STATE_LABELS)


if __name__ == "__main__":
    main()