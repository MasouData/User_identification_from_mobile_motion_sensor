# Databricks notebook source
# MAGIC %pip install xgboost

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
  "percentile_approx(count, 0.5) as p50",
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
]

for c in fields:
  agg_exprs += [
    F.avg(c).alias(f"{c}_mean"),
    F.stddev(c).alias(f"{c}_std"),
    F.min(c).alias(f"{c}_min"),
    F.max(c).alias(f"{c}_max"),
    F.count(c).alias(f"{c}_nn")   # non-null count
  ]

def clean_col(colname):
    import re
    m = re.match(r"^(\d+)_first\((.+)\)$", colname)
    if m:
        sensor, metric = m.group(1), m.group(2)
        return f"sensor_{sensor}_{metric}"
    return colname

def build_session_features(events_df, has_label: bool):
    # aggregate per session x sensor
    sess_sensor = events_df.groupBy("session_id", "sensor_type").agg(*agg_exprs)
    metric_cols = [c for c in sess_sensor.columns if c not in ("session_id", "sensor_type")]

    # pivot sensor -> wide columns
    wide = (
        sess_sensor.groupBy("session_id")
                   .pivot("sensor_type")
                   .agg(*[F.first(c) for c in metric_cols])
    )

    # clean pivoted column names
    for c in wide.columns:
        wide = wide.withColumnRenamed(c, clean_col(c))

    if has_label:
        labels = events_df.select("session_id", "user_id").distinct()
        wide = wide.join(labels, on="session_id", how="left")
    return wide

train_features = build_session_features(train_df, has_label=True)
test_features  = build_session_features(test_df, has_label=False)

# Align columns train/test and fill missing with 0
train_cols = set(train_features.columns) - {"user_id"}
test_cols  = set(test_features.columns)
all_feature_cols = sorted((train_cols | test_cols) - {"session_id"})

for c in all_feature_cols:
    if c not in train_features.columns:
        train_features = train_features.withColumn(c, F.lit(None).cast("double"))
    if c not in test_features.columns:
        test_features = test_features.withColumn(c, F.lit(None).cast("double"))

train_features = train_features.select(["session_id"] + all_feature_cols + ["user_id"]).fillna(0)
test_features  = test_features.select(["session_id"] + all_feature_cols).fillna(0)

print("train sessions:", train_features.count(), "test sessions:", test_features.count())

# COMMAND ----------

# see relevant imbalance
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

pdf_train = train_features.toPandas()
pdf_test  = test_features.toPandas()

X = pdf_train.drop(columns=["session_id", "user_id"])
y = pdf_train["user_id"]
X_test = pdf_test.drop(columns=["session_id"])

# COMMAND ----------

#Model comparison: F1

def repeated_eval_macro_f1(X, y, seeds=(0,1,2,3,4)):
    out = {}
    # XGB needs integer labels
    le = LabelEncoder()
    y_enc_all = le.fit_transform(y)

    out["Dummy_most_frequent"] = []
    out["Dummy_stratified"] = []
    out["LR"] = []
    out["SVM"] = []
    out["RF"] = []
    out["XGB"] = []


    for seed in seeds:
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )

        dummy_mf = DummyClassifier(strategy="most_frequent", random_state=seed).fit(X_tr, y_tr)
        out["Dummy_most_frequent"].append(f1_score(y_va, dummy_mf.predict(X_va), average="macro"))

        dummy_st = DummyClassifier(strategy="stratified", random_state=seed).fit(X_tr, y_tr)
        out["Dummy_stratified"].append(f1_score(y_va, dummy_st.predict(X_va), average="macro"))

        lr = Pipeline([("scaler", StandardScaler()),
                       ("lr", LogisticRegression(max_iter=3000, C=1.0, n_jobs=-1))]).fit(X_tr, y_tr)
        out["LR"].append(f1_score(y_va, lr.predict(X_va), average="macro"))

        svm = Pipeline([("scaler", StandardScaler()),
                        ("svm", LinearSVC(C=1.0))]).fit(X_tr, y_tr)
        out["SVM"].append(f1_score(y_va, svm.predict(X_va), average="macro"))

        rf = RandomForestClassifier(
            n_estimators=500,
            random_state=seed,
            class_weight="balanced_subsample"
        ).fit(X_tr, y_tr)
        out["RF"].append(f1_score(y_va, rf.predict(X_va), average="macro"))

        # XGB with encoded labels on same seed
        X_tr2, X_va2, y_tr2, y_va2 = train_test_split(
            X, y_enc_all, test_size=0.2, random_state=seed, stratify=y_enc_all
        )
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
        ).fit(X_tr2, y_tr2)
        out["XGB"].append(f1_score(y_va2, xgb.predict(X_va2), average="macro"))

    print("Macro-F1 means (5 seeds):")
    for k, vals in out.items():
        print(f"{k:22s} mean={np.mean(vals):.4f} scores={vals}")
    return out

_ = repeated_eval_macro_f1(X, y)

# COMMAND ----------

#RF tuning
seeds = [0, 1, 2, 3, 4]
grid = [
    {"n_estimators": 300, "max_depth": None, "max_features": "sqrt"},
    {"n_estimators": 600, "max_depth": None, "max_features": "sqrt"},
    {"n_estimators": 600, "max_depth": 20,   "max_features": "sqrt"},
    {"n_estimators": 600, "max_depth": 10,   "max_features": "sqrt"},
]
leaf_grid = [1, 2, 5]

results = []
for base_params, leaf in product(grid, leaf_grid):
    params = dict(base_params)
    params["min_samples_leaf"] = leaf

    scores = []
    for seed in seeds:
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        rf = RandomForestClassifier(
            random_state=seed,
            class_weight="balanced_subsample",
            **params
        ).fit(X_tr, y_tr)

        scores.append(f1_score(y_va, rf.predict(X_va), average="macro"))

    results.append({"params": params, "mean": float(np.mean(scores)), "scores": scores})

results_sorted = sorted(results, key=lambda d: d["mean"], reverse=True)
print("Top RF configs:")
for r in results_sorted[:5]:
    print(r["params"], "mean=", r["mean"], "scores=", r["scores"])

best_params = results_sorted[0]["params"]
print("Selected best_params:", best_params)

# COMMAND ----------

rf_final = RandomForestClassifier(
    random_state=42,
    class_weight="balanced_subsample",
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
