"""Validate recovery of the latent Intent-to-Return variable."""
import numpy as np
from sklearn.metrics import roc_auc_score
from psf import (
    PSFModel, simulate_cohort, baum_welch,
    forward_backward, intent_posterior,
)
from psf.states import IDX
from examples._common import build_true_model  # assume shared helper


def main():
    true_model = build_true_model()
    data = simulate_cohort(true_model, n_students=400, max_time=36)

    model = PSFModel(n_features=6)
    baum_welch(model, data["observations"], data["masks"], n_iter=30)

    all_intent, all_true = [], []
    for seq, m, true_states in zip(
        data["observations"], data["masks"], data["states"]
    ):
        out = forward_backward(model, seq, m)
        intent = intent_posterior(out["gamma"])
        true_intent = (true_states == IDX.S3_INTENT).astype(float)
        s3_mask = (true_states == IDX.S3_NO_INTENT) | (true_states == IDX.S3_INTENT)
        all_intent.extend(intent[s3_mask])
        all_true.extend(true_intent[s3_mask])

    all_intent = np.nan_to_num(all_intent)
    auc = roc_auc_score(all_true, all_intent)
    print(f"Intent-to-Return AUC: {auc:.3f}")


if __name__ == "__main__":
    main()