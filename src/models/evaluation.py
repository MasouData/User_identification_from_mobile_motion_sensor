# src/models/evaluation.py

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon


def summarize_model_scores(results):
    """
    Summarize repeated Macro-F1 scores for every model.

    Parameters
    ----------
    results : dict
        Model name -> list of Macro-F1 scores.

    Returns
    -------
    pandas.DataFrame
        Mean, standard deviation, minimum and maximum
        Macro-F1 for each model.
    """

    rows = []

    for model_name, scores in results.items():

        scores_array = np.asarray(
            scores,
            dtype=float,
        )

        rows.append({
            "model": model_name,
            "mean_macro_f1": scores_array.mean(),
            "std_macro_f1": scores_array.std(),
            "min_macro_f1": scores_array.min(),
            "max_macro_f1": scores_array.max(),
        })

    summary = pd.DataFrame(rows)

    return summary.sort_values(
        "mean_macro_f1",
        ascending=False,
    ).reset_index(drop=True)


def paired_wilcoxon_test(
    scores_a,
    scores_b,
    model_a="RF",
    model_b="XGB",
):
    """
    Compare two models evaluated on exactly the same
    repeated train/validation splits.

    Uses the paired Wilcoxon signed-rank test.

    Returns
    -------
    dict
        Summary statistics and p-value.
    """

    scores_a = np.asarray(
        scores_a,
        dtype=float,
    )

    scores_b = np.asarray(
        scores_b,
        dtype=float,
    )

    if len(scores_a) != len(scores_b):
        raise ValueError(
            "Paired model score arrays must have the same length."
        )

    differences = (
        scores_a - scores_b
    )

    statistic, p_value = wilcoxon(
        differences
    )

    return {
        "model_a": model_a,
        "model_b": model_b,
        "mean_a": float(
            scores_a.mean()
        ),
        "mean_b": float(
            scores_b.mean()
        ),
        "mean_difference": float(
            differences.mean()
        ),
        "wilcoxon_statistic": float(
            statistic
        ),
        "p_value": float(
            p_value
        ),
        "significant_at_0_05": bool(
            p_value < 0.05
        ),
    }