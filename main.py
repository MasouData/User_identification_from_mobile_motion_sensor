# Databricks notebook source
# MAGIC %pip install -r requirements.txt

# COMMAND ----------

from src.cli import (
    load_config,
    run_experiment,
)
config = load_config(
    "configs/baseline.yaml"
)

results = run_experiment(
    config,
    spark=spark,
)

# COMMAND ----------

display(
    results["model_summary"]
)

# COMMAND ----------

display(
    results["submission"].head()
)

# COMMAND ----------

from src.data.loading import (
    load_train_data,
    load_test_data,
)

from src.data.validation import (
    get_null_counts,
    find_mixed_user_sessions,
    get_session_size_stats,
    get_sensor_types,
    get_user_session_counts,
    get_imbalance_ratio,
)

from src.features.session_features import (
    build_session_features,
    align_train_test_features,
)



# ---------------------------------------------------------
# Configuration - temporary
#
# We will move these paths to baseline.yaml later.
# ---------------------------------------------------------

train_path = "/Volumes/workspace/threatfabric/project/train.csv"

test_path = "/Volumes/workspace/threatfabric/project/test.csv"



# ---------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------

train_df = load_train_data(
    spark,
    train_path
)

test_df = load_test_data(
    spark,
    test_path
)

print("Train rows:", train_df.count())
print("Test rows:", test_df.count())

display(
    train_df.limit(5)
)



# ---------------------------------------------------------
# 2. Data validation
# ---------------------------------------------------------

print("TRAIN NULL COUNTS")
display(
    get_null_counts(train_df)
)


print("TEST NULL COUNTS")
display(
    get_null_counts(test_df)
)


# Mixed-user session check
mixed_sessions = (
    find_mixed_user_sessions(
        train_df
    )
)

mixed_count = mixed_sessions.count()

print(
    "Mixed-user sessions:",
    mixed_count
)

if mixed_count > 0:
    display(
        mixed_sessions
    )


# Session size distribution
print(
    "SESSION SIZE STATISTICS"
)

display(
    get_session_size_stats(
        train_df
    )
)


# Sensor types
print(
    "SENSOR TYPES"
)

display(
    get_sensor_types(
        train_df
    )
)


# Sessions per user
print(
    "SESSIONS PER USER"
)

display(
    get_user_session_counts(
        train_df
    )
)


# Class imbalance
imbalance_ratio = (
    get_imbalance_ratio(
        train_df
    )
)

print(
    "Class imbalance ratio:",
    imbalance_ratio
)



# ---------------------------------------------------------
# 3. Session-level feature engineering
# ---------------------------------------------------------

train_features = (
    build_session_features(
        train_df,
        has_label=True
    )
)

test_features = (
    build_session_features(
        test_df,
        has_label=False
    )
)



# ---------------------------------------------------------
# 4. Align train/test features
# ---------------------------------------------------------

(
    train_features,
    test_features,
    feature_columns,
) = align_train_test_features(
    train_features,
    test_features,
)



# ---------------------------------------------------------
# 5. Verify output
# ---------------------------------------------------------

print(
    "Train sessions:",
    train_features.count()
)

print(
    "Test sessions:",
    test_features.count()
)

print(
    "Train columns:",
    len(train_features.columns)
)

print(
    "Test columns:",
    len(test_features.columns)
)

print(
    "Number of ML features:",
    len(feature_columns)
)


display(
    train_features.limit(5)
)

# COMMAND ----------

# MAGIC %pip install xgboost

# COMMAND ----------

from src.models.training import (
    compare_models_repeated,
    tune_random_forest,
    train_final_random_forest,
)

from src.models.evaluation import (
    summarize_model_scores,
    paired_wilcoxon_test,
)

import pandas as pd


# ---------------------------------------------------------
# Convert session-level Spark data to Pandas
# ---------------------------------------------------------

pdf_train = train_features.toPandas()
pdf_test = test_features.toPandas()


X = pdf_train.drop(
    columns=[
        "session_id",
        "user_id",
    ]
)

y = pdf_train["user_id"]


X_test = pdf_test.drop(
    columns=[
        "session_id",
    ]
)


print(
    "X shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)

print(
    "X_test shape:",
    X_test.shape
)

# COMMAND ----------

# DBTITLE 1,smoke test
seeds = list(
    range(20)
)

model_results = compare_models_repeated(
    X,
    y,
    seeds=seeds,
)


model_summary = summarize_model_scores(
    model_results
)


display(
    model_summary
)

# COMMAND ----------

# DBTITLE 1,Statistically compare RF vs XGB
rf_xgb_comparison = paired_wilcoxon_test(
    model_results["RF"],
    model_results["XGB"],
    model_a="Random Forest",
    model_b="XGBoost",
)


print(
    "RF mean Macro-F1:",
    rf_xgb_comparison["mean_a"]
)

print(
    "XGB mean Macro-F1:",
    rf_xgb_comparison["mean_b"]
)

print(
    "Mean difference (RF - XGB):",
    rf_xgb_comparison["mean_difference"]
)

print(
    "Wilcoxon statistic:",
    rf_xgb_comparison["wilcoxon_statistic"]
)

print(
    "p-value:",
    rf_xgb_comparison["p_value"]
)

print(
    "Significant at 0.05:",
    rf_xgb_comparison["significant_at_0_05"]
)

# COMMAND ----------

# DBTITLE 1,tune RF
best_parameters, rf_tuning_results = (
    tune_random_forest(
        X,
        y,
        seeds=seeds,
    )
)


print(
    "Selected RF parameters:"
)

print(
    best_parameters
)


print(
    "\nTop RF configurations:"
)

for result in rf_tuning_results[:5]:

    print(
        result["params"],
        "mean=",
        round(
            result["mean"],
            4
        ),
        "std=",
        round(
            result["std"],
            4
        ),
    )

# COMMAND ----------

# DBTITLE 1,train final model and generate submission
final_model = train_final_random_forest(
    X,
    y,
    best_parameters,
)


test_predictions = final_model.predict(
    X_test
)


submission = pd.DataFrame({
    "session_id": pdf_test["session_id"],
    "user_id": test_predictions,
})


submission.to_csv(
    "submission.csv",
    index=False,
)


print(
    "Saved submission.csv rows:",
    len(submission)
)


display(
    submission.head()
)

# COMMAND ----------

# DBTITLE 1,visualization
from src.visualization.plots import (
    plot_model_comparison,
)


fig = plot_model_comparison(
    model_summary,
    output_path="images/model_comparison.png",
)

display(fig)

# COMMAND ----------

# import sys
# import os

# # Disable Python bytecode generation (Workspace filesystem doesn't support __pycache__)
# sys.dont_write_bytecode = True
# os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

# import pytest

# project_root = "/Workspace/Users/masoud_aghayan200@yahoo.com/Data_Science_Assignment"

# # Add project root to path so tests can import from src
# sys.path.insert(0, project_root)

# exit_code = pytest.main([
#     "-v",
#     "-p", "no:cacheprovider",
#     f"{project_root}/tests",
# ])

# assert exit_code == 0
