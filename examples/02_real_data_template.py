"""Template for running the PSF on institutional data.

Replace `load_data()` with your institutional data loader. It should return:
    records : list[StudentRecord]
    observations : list[(T_i, D) np.ndarray]
    masks : list[(T_i, D) bool np.ndarray]
"""
import numpy as np
from psf import (
    PSFModel, baum_welch, forward_backward,
    generate_operational_labels, evaluate_predictions,
)


def load_data():
    """Load institutional data.

    Returns
    -------
    records : list[StudentRecord]
    observations : list[np.ndarray]
    masks : list[np.ndarray]
    """
    raise NotImplementedError(
        "Replace this with your institutional data loader. "
        "See docs/user_guide.md Section 5 for details."
    )


def main():
    records, obs, masks = load_data()

    labels = generate_operational_labels(records, W=24, T_pause=2)
    print(f"Label distribution: "
          f"Completed={np.sum(labels==0)}, "
          f"Withdrawn={np.sum(labels==1)}, "
          f"Stopped-Out={np.sum(labels==2)}, "
          f"Censored={np.sum(labels==3)}")

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(obs))
    split = int(0.7 * len(idx))
    train_idx, test_idx = idx[:split], idx[split:]

    model = PSFModel(n_features=obs[0].shape[1])
    baum_welch(
        model,
        [obs[i] for i in train_idx],
        [masks[i] for i in train_idx],
        n_iter=50,
        verbose=True,
    )

    y_test = (labels[test_idx] == 1).astype(int)
    psf_prob = []
    for i in test_idx:
        out = forward_backward(model, obs[i], masks[i], return_xi=False)
        g = out["gamma"][-1]
        psf_prob.append(g[3] + g[2])
    psf_prob = np.asarray(psf_prob)

    metrics = evaluate_predictions(y_test, psf_prob)
    for k, v in metrics.items():
        print(f"{k:12s}: {v:.4f}")


if __name__ == "__main__":
    main()