import numpy as np
import pandas as pd
import pytest

from src.models.training import (
    train_final_random_forest,
)

from src.models.evaluation import (
    summarize_model_scores,
)


def test_final_random_forest_can_fit_and_predict():
    """
    The final Random Forest pipeline should train
    successfully and return one prediction per row.
    """

    rng = np.random.default_rng(
        42
    )

    # Clearly separated synthetic classes.
    class_a = rng.normal(
        loc=0.0,
        scale=0.2,
        size=(10, 3),
    )

    class_b = rng.normal(
        loc=5.0,
        scale=0.2,
        size=(10, 3),
    )

    X = pd.DataFrame(
        np.vstack([
            class_a,
            class_b,
        ]),
        columns=[
            "feature_1",
            "feature_2",
            "feature_3",
        ],
    )

    y = pd.Series(
        ["user_a"] * 10
        + ["user_b"] * 10
    )

    parameters = {
        "n_estimators": 20,
        "max_depth": None,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
    }

    model = train_final_random_forest(
        X,
        y,
        best_parameters=parameters,
        random_state=42,
    )

    predictions = model.predict(
        X
    )

    assert len(predictions) == len(y)

    assert set(predictions).issubset(
        {"user_a", "user_b"}
    )

    training_accuracy = (
        predictions == y
    ).mean()

    assert training_accuracy >= 0.95


def test_model_score_summary():
    """
    Model-score summaries should calculate the
    correct ordering and mean Macro-F1.
    """

    results = {
        "RF": [
            0.90,
            0.95,
            1.00,
        ],

        "XGB": [
            0.80,
            0.85,
            0.90,
        ],
    }

    summary = (
        summarize_model_scores(
            results
        )
    )

    # RF should rank first.
    assert (
        summary.iloc[0]["model"]
        == "RF"
    )

    rf_row = (
        summary[
            summary["model"]
            == "RF"
        ]
        .iloc[0]
    )

    assert (
        rf_row["mean_macro_f1"]
        == pytest.approx(0.95)
    )