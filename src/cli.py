import argparse
from pathlib import Path

import pandas as pd
import yaml

from pyspark.sql import SparkSession

from src.data.loading import (
    load_train_data,
    load_test_data,
)

from src.data.validation import (
    find_mixed_user_sessions,
    get_imbalance_ratio,
)

from src.features.session_features import (
    build_session_features,
    align_train_test_features,
)

from src.models.training import (
    compare_models_repeated,
    tune_random_forest,
    train_final_random_forest,
)

from src.models.evaluation import (
    summarize_model_scores,
    paired_wilcoxon_test,
)

from src.visualization.plots import (
    plot_model_comparison,
)


def load_config(config_path):
    """
    Load experiment configuration from a YAML file.
    """

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    return config


def run_experiment(
    config,
    spark=None,
):
    """
    Run the complete motion-based user-identification experiment.

    The workflow performs:

    1. Data loading
    2. Data-quality validation
    3. Session-level feature engineering
    4. Train/test feature alignment
    5. Repeated model comparison
    6. RF vs XGB statistical comparison
    7. Random Forest tuning
    8. Final model training
    9. Test prediction
    10. Result visualization

    Parameters
    ----------
    config : dict
        Configuration loaded from baseline.yaml.

    spark : SparkSession, optional
        Existing Spark session. In Databricks we pass the
        already available Spark session.

    Returns
    -------
    dict
        Important experiment outputs.
    """

    # ---------------------------------------------------------
    # Spark
    # ---------------------------------------------------------

    if spark is None:
        spark = (
            SparkSession.builder
            .getOrCreate()
        )

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    train_path = (
        config["data"]["train_path"]
    )

    test_path = (
        config["data"]["test_path"]
    )

    n_seeds = int(
        config["evaluation"]["n_seeds"]
    )

    random_state = int(
        config["model"]["random_state"]
    )

    submission_path = Path(
        config["output"]["submission_path"]
    )

    figure_path = Path(
        config["output"]["model_comparison_path"]
    )

    seeds = list(
        range(n_seeds)
    )

    print(
        "\n=========================================="
    )

    print(
        config["project"]["name"]
    )

    print(
        "=========================================="
    )

    # ---------------------------------------------------------
    # 1. Load raw sensor data
    # ---------------------------------------------------------

    print(
        "\n[1/8] Loading data..."
    )

    train_df = load_train_data(
        spark,
        train_path,
    )

    test_df = load_test_data(
        spark,
        test_path,
    )

    train_rows = train_df.count()
    test_rows = test_df.count()

    print(
        f"Train rows: {train_rows}"
    )

    print(
        f"Test rows:  {test_rows}"
    )

    # ---------------------------------------------------------
    # 2. Validate training data
    # ---------------------------------------------------------

    print(
        "\n[2/8] Validating data..."
    )

    mixed_sessions = (
        find_mixed_user_sessions(
            train_df
        )
    )

    mixed_session_count = (
        mixed_sessions.count()
    )

    print(
        "Mixed-user sessions:",
        mixed_session_count,
    )

    if mixed_session_count > 0:
        raise ValueError(
            "Data validation failed: "
            "a session belongs to more than one user."
        )

    imbalance_ratio = (
        get_imbalance_ratio(
            train_df
        )
    )

    print(
        "Class imbalance ratio:",
        imbalance_ratio,
    )

    # ---------------------------------------------------------
    # 3. Feature engineering
    # ---------------------------------------------------------

    print(
        "\n[3/8] Building session features..."
    )

    train_features = (
        build_session_features(
            train_df,
            has_label=True,
        )
    )

    test_features = (
        build_session_features(
            test_df,
            has_label=False,
        )
    )

    (
        train_features,
        test_features,
        feature_columns,
    ) = align_train_test_features(
        train_features,
        test_features,
    )

    train_sessions = (
        train_features.count()
    )

    test_sessions = (
        test_features.count()
    )

    print(
        "Train sessions:",
        train_sessions,
    )

    print(
        "Test sessions:",
        test_sessions,
    )

    print(
        "Number of ML features:",
        len(feature_columns),
    )

    # ---------------------------------------------------------
    # 4. Convert session-level data to Pandas
    # ---------------------------------------------------------

    print(
        "\n[4/8] Preparing ML matrices..."
    )

    pdf_train = (
        train_features.toPandas()
    )

    pdf_test = (
        test_features.toPandas()
    )

    X = pdf_train.drop(
        columns=[
            "session_id",
            "user_id",
        ]
    )

    y = pdf_train[
        "user_id"
    ]

    X_test = pdf_test.drop(
        columns=[
            "session_id",
        ]
    )

    print(
        "X shape:",
        X.shape,
    )

    print(
        "y shape:",
        y.shape,
    )

    print(
        "X_test shape:",
        X_test.shape,
    )

    # ---------------------------------------------------------
    # 5. Model comparison
    # ---------------------------------------------------------

    print(
        "\n[5/8] Comparing models..."
    )

    model_results = (
        compare_models_repeated(
            X,
            y,
            seeds=seeds,
        )
    )

    model_summary = (
        summarize_model_scores(
            model_results
        )
    )

    print(
        "\nModel comparison:"
    )

    print(
        model_summary.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # 6. Statistical comparison
    # ---------------------------------------------------------

    print(
        "\n[6/8] Comparing RF and XGB..."
    )

    rf_xgb_comparison = (
        paired_wilcoxon_test(
            model_results["RF"],
            model_results["XGB"],
            model_a="Random Forest",
            model_b="XGBoost",
        )
    )

    print(
        "RF mean Macro-F1:",
        rf_xgb_comparison["mean_a"],
    )

    print(
        "XGB mean Macro-F1:",
        rf_xgb_comparison["mean_b"],
    )

    print(
        "RF - XGB difference:",
        rf_xgb_comparison[
            "mean_difference"
        ],
    )

    print(
        "Wilcoxon p-value:",
        rf_xgb_comparison[
            "p_value"
        ],
    )

    # ---------------------------------------------------------
    # Generate README visualization
    # ---------------------------------------------------------

    figure_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure = (
        plot_model_comparison(
            model_summary,
            output_path=figure_path,
        )
    )

    print(
        f"Saved figure: {figure_path}"
    )

    # ---------------------------------------------------------
    # 7. Tune and train final RF
    # ---------------------------------------------------------

    print(
        "\n[7/8] Tuning Random Forest..."
    )

    (
        best_parameters,
        tuning_results,
    ) = tune_random_forest(
        X,
        y,
        seeds=seeds,
    )

    print(
        "Best RF parameters:"
    )

    print(
        best_parameters
    )

    final_model = (
        train_final_random_forest(
            X,
            y,
            best_parameters,
            random_state=random_state,
        )
    )

    # ---------------------------------------------------------
    # 8. Predict test sessions
    # ---------------------------------------------------------

    print(
        "\n[8/8] Generating predictions..."
    )

    predictions = (
        final_model.predict(
            X_test
        )
    )

    submission = pd.DataFrame({
        "session_id":
            pdf_test["session_id"],

        "user_id":
            predictions,
    })

    submission_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    submission.to_csv(
        submission_path,
        index=False,
    )

    print(
        f"Saved {len(submission)} predictions "
        f"to {submission_path}"
    )

    print(
        "\nExperiment completed successfully."
    )

    # ---------------------------------------------------------
    # Return useful objects for notebooks/tests
    # ---------------------------------------------------------

    return {
        "model_summary":
            model_summary,

        "rf_xgb_comparison":
            rf_xgb_comparison,

        "best_parameters":
            best_parameters,

        "tuning_results":
            tuning_results,

        "submission":
            submission,

        "figure":
            figure,

        "n_features":
            len(feature_columns),

        "train_sessions":
            train_sessions,

        "test_sessions":
            test_sessions,
    }


def main():
    """
    Command-line entry point.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run the reproducible motion-based "
            "user-identification experiment."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help=(
            "Path to the YAML experiment "
            "configuration file."
        ),
    )

    args = parser.parse_args()

    config = load_config(
        args.config
    )

    run_experiment(
        config
    )


if __name__ == "__main__":
    main()