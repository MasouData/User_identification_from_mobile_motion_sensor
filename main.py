# Databricks notebook source
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql.types import *
from itertools import product

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier

# COMMAND ----------

train_path = "/Volumes/workspace/threatfabric/project/train.csv"
test_path  = "/Volumes/workspace/threatfabric/project/test.csv"

schema_train = StructType([
  StructField("uid", StringType()),
  StructField("timestamp", LongType()),
  StructField("sensor_type", StringType()),
  *[StructField(f"field_{i}", DoubleType()) for i in range(8)],
  StructField("session_id", StringType()),
  StructField("user_id", StringType()),
])

schema_test = StructType([f for f in schema_train.fields if f.name != "user_id"])

train_df = (spark.read.option("header", True).schema(schema_train).csv(train_path))
test_df  = (spark.read.option("header", True).schema(schema_test).csv(test_path))

train_df.limit(5).display()

# COMMAND ----------

display(train_df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in train_df.columns]))

bad_sessions = (
  train_df.groupBy("session_id")
          .agg(F.countDistinct("user_id").alias("n_users"))
          .filter("n_users > 1")
)

print("bad_sessions:", bad_sessions.count())

session_sizes = train_df.groupBy("session_id").count()
display(session_sizes.selectExpr(
  " percentile_approx(count, 0.5) as p50",
  "percentile_approx(count, 0.9) as p90",
  "percentile_approx(count, 0.99) as p99",
  "max(count) as max_count"
))
# Sensor types
display(train_df.select("sensor_type").distinct().orderBy("sensor_type"))

# COMMAND ----------

# Feature engineering -  one row per session
fields = [f"field_{i}" for i in range(8)]

agg_exprs = [
  F.count("*").alias("n_events"),
  (F.max("timestamp") - F.min("timestamp")).alias("duration_ms"),
]

# per-field stats
for c in fields:
  agg_exprs += [
    F.avg(c).alias(f"{c}_mean"),
    F.expr(f"percentile_approx({c}, 0.5)").alias(f"{c}_p50"),  # median
    F.stddev(c).alias(f"{c}_std"),
    F.min(c).alias(f"{c}_min"),
    F.max(c).alias(f"{c}_max"),
    F.count(c).alias(f"{c}_nn")   # non-null count
  ]

# magnitude stats (for sensors with axes in field_0..field_2)
agg_exprs += [
    F.avg("mag_012").alias("mag_mean"),
    F.expr("percentile_approx(mag_012, 0.5)").alias("mag_p50"),  # median magnitude
    F.stddev("mag_012").alias("mag_std"),
    F.min("mag_012").alias("mag_min"),
    F.max("mag_012").alias("mag_max"),
    F.count("mag_012").alias("mag_nn"),
]

def clean_col(colname):
    import re
    m = re.match(r"^(\d+)_first\((.+)\)$", colname)
    if m:
        sensor, metric = m.group(1), m.group(2)
        return f"sensor_{sensor}_{metric}"
    return colname

def build_session_features(events_df, has_label):
    # Add magnitude of xyz axes
    events_df = events_df.withColumn(
        "mag_012",
        F.sqrt(
            F.col("field_0")*F.col("field_0") +
            F.col("field_1")*F.col("field_1") +
            F.col("field_2")*F.col("field_2")
        )
    )
    sess_sensor = events_df.groupBy("session_id", "sensor_type").agg(*agg_exprs)

    sess_sensor = sess_sensor.withColumn(
        "event_rate",
        F.col("n_events") / (F.col("duration_ms") + F.lit(1.0))
    )

    metric_cols = [c for c in sess_sensor.columns if c not in ("session_id", "sensor_type")]

    # pivot sensor -> wide columns. (sensor_1_field_0_mean, sensor_2_mag_std, sensor_6_field_7_max)
    wide = (
        sess_sensor.groupBy("session_id")
                   .pivot("sensor_type")
                   .agg(*[F.first(c) for c in metric_cols])
    )

    # clean pivoted column names
    for c in wide.columns:
        wide = wide.withColumnRenamed(c, clean_col(c))
        
    #Add user_id label only for train
    if has_label:
        labels = events_df.select("session_id", "user_id").distinct()
        wide = wide.join(labels, on="session_id", how="left")
    return wide

train_features = build_session_features(train_df, has_label=True)
test_features  = build_session_features(test_df, has_label=False)

# Align columns train/test and fill missing with Null. Num of features/columns in both train and test MUST be same.  
train_cols = set(train_features.columns) - {"user_id"}
test_cols  = set(test_features.columns)
all_feature_cols = sorted((train_cols | test_cols) - {"session_id"})

#If a sensor not appeared in test/testt, it adds the same feature as NULL
for c in all_feature_cols:
    if c not in train_features.columns:
        train_features = train_features.withColumn(c, F.lit(None).cast("double"))
    if c not in test_features.columns:
        test_features = test_features.withColumn(c, F.lit(None).cast("double"))

train_features = train_features.select(["session_id"] + all_feature_cols + ["user_id"]).fillna(0)
test_features  = test_features.select(["session_id"] + all_feature_cols).fillna(0)

print("train sessions:", train_features.count(), "test sessions:", test_features.count())
print("train features:", len(train_features.columns), "test features:", len(test_features.columns))

# COMMAND ----------

# see relevant imbalance/ For each user there are 15 sessons => Balance
sess_labels = train_df.select("session_id", "user_id").distinct()
sess_per_user = sess_labels.groupBy("user_id").count().orderBy(F.desc("count"))
spu = sess_per_user.toPandas()

plt.figure()
plt.bar(range(len(spu)), spu["count"])
plt.xticks(range(len(spu)), spu["user_id"], rotation=90)
plt.ylabel("# sessions per user")
plt.title("Class imbalance (sessions per user)")
plt.tight_layout()
plt.show()

# COMMAND ----------

# Check actual imbalance => Balance
user_session_counts = train_df.groupBy("user_id").agg(
    F.countDistinct("session_id").alias("n_sessions")
).toPandas()

print("Session count stats:", user_session_counts['n_sessions'].describe())
print("Imbalance ratio:", user_session_counts['n_sessions'].max() / 
                          user_session_counts['n_sessions'].min())

# COMMAND ----------

#High number of features => Cannot use Databricks Free Edition => Switch to Pandas
pdf_train = train_features.toPandas()
pdf_test  = test_features.toPandas()

X = pdf_train.drop(columns=["session_id", "user_id"])
y = pdf_train["user_id"]
X_test = pdf_test.drop(columns=["session_id"])

# COMMAND ----------

#Model comparison: F1
def repeated_eval_macro_f1_same_split(X, y, seeds=(0,1,2,3,4,42)):
    out = {
        "Dummy_most_frequent": [],
        "Dummy_stratified": [],
        "LR": [],
        "SVM": [],
        "RF": [],
        "XGB": []
    }

    # Encode once for XGB
    le = LabelEncoder()
    y_enc_all = le.fit_transform(y)

    # Create an index array once
    idx = np.arange(len(y))

    for seed in seeds:
        # Split indices ONCE (stratify using y)
        tr_idx, va_idx = train_test_split(
            idx, test_size=0.2, random_state=seed, stratify=y
        )

        # Slice X and both label versions using the SAME indices
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        y_tr_enc, y_va_enc = y_enc_all[tr_idx], y_enc_all[va_idx]

        # ---- Baselines + models using string labels ----
        dummy_mf = DummyClassifier(strategy="most_frequent", random_state=seed).fit(X_tr, y_tr)
        out["Dummy_most_frequent"].append(f1_score(y_va, dummy_mf.predict(X_va), average="macro"))

        dummy_st = DummyClassifier(strategy="stratified", random_state=seed).fit(X_tr, y_tr)
        out["Dummy_stratified"].append(f1_score(y_va, dummy_st.predict(X_va), average="macro"))

        lr = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=3000, C=1.0, n_jobs=-1))
        ]).fit(X_tr, y_tr)
        out["LR"].append(f1_score(y_va, lr.predict(X_va), average="macro"))

        svm = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", LinearSVC(C=1.0))
        ]).fit(X_tr, y_tr)
        out["SVM"].append(f1_score(y_va, svm.predict(X_va), average="macro"))

        rf = RandomForestClassifier(
            n_estimators=500,
            random_state=seed
        ).fit(X_tr, y_tr)
        out["RF"].append(f1_score(y_va, rf.predict(X_va), average="macro"))

        xgb = XGBClassifier(
            objective="multi:softmax",
            num_class=len(le.classes_),
            eval_metric="mlogloss",
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed
        ).fit(X_tr, y_tr_enc)

        out["XGB"].append(f1_score(y_va_enc, xgb.predict(X_va), average="macro"))

    print("Macro-F1 means (same splits):")
    for k, vals in out.items():
        print(f"{k:22s} mean={np.mean(vals):.4f} scores={vals}")

    return out
seeds = list(range(20))    
res = repeated_eval_macro_f1_same_split(X, y, seeds)
rf_scores  = res["RF"]
xgb_scores = res["XGB"]

# COMMAND ----------

from scipy.stats import wilcoxon   #non-parametric paired test

print("RF mean/std:", rf.mean(), rf.std())
print("XGB mean/std:", xgb.mean(), xgb.std())

rf = np.array(rf_scores)   # your list from same-split evaluation
xgb = np.array(xgb_scores)

diff = rf - xgb
print("Mean diff (RF - XGB):", diff.mean())

stat, p = wilcoxon(diff)
print("Wilcoxon p-value:", p)    #p >= 0.05 → “not statistically significant” 

# COMMAND ----------

# RF => is final model 
rf_grid = [
    {"n_estimators": 300, "max_depth": None, "max_features": "sqrt", "min_samples_leaf": 1},
    {"n_estimators": 500, "max_depth": None, "max_features": "sqrt", "min_samples_leaf": 1},
    {"n_estimators": 500, "max_depth": 20,   "max_features": "sqrt", "min_samples_leaf": 1},
    {"n_estimators": 500, "max_depth": 10,   "max_features": "sqrt", "min_samples_leaf": 1},
    {"n_estimators": 500, "max_depth": None, "max_features": "sqrt", "min_samples_leaf": 2},
    {"n_estimators": 500, "max_depth": None, "max_features": "sqrt", "min_samples_leaf": 5},
]

results = []

for params in rf_grid:
    scores = []
    for seed in seeds:
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )

        rf = RandomForestClassifier(
            random_state=seed,
            class_weight="balanced_subsample",
            n_jobs=-1,
            **params
        )
        rf.fit(X_tr, y_tr)
        preds = rf.predict(X_va)
        scores.append(f1_score(y_va, preds, average="macro"))

    results.append({
        "params": params,
        "mean": float(np.mean(scores)),
        "std":  float(np.std(scores)),
        "scores": scores
    })

results_sorted = sorted(results, key=lambda d: d["mean"], reverse=True)

print("Top RF configs:")
for r in results_sorted[:5]:
    print(r["params"], "mean=", r["mean"], "std=", r["std"])

best_params = results_sorted[0]["params"]
print("Selected best_params:", best_params)

# COMMAND ----------

# Train final model on all training sessions
rf_final = RandomForestClassifier(
    random_state=42,
    class_weight="balanced_subsample",
    n_jobs=-1,
    **best_params
)

rf_final.fit(X, y)

test_pred = rf_final.predict(X_test)

submission = pd.DataFrame({
    "session_id": pdf_test["session_id"],
    "user_id": test_pred
})

submission.to_csv("submission.csv", index=False)
print("Saved submission.csv rows:", len(submission))
display(submission.head())

