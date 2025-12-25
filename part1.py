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


# COMMAND ----------

train_df.limit(5).display()

# COMMAND ----------

train_df.printSchema()
test_df.printSchema()

display(train_df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in train_df.columns]))


# COMMAND ----------

bad_sessions = (
  train_df.groupBy("session_id")
          .agg(F.countDistinct("user_id").alias("n_users"))
          .filter("n_users > 1")
)

bad_sessions.count()

# COMMAND ----------

session_sizes = train_df.groupBy("session_id").count()
display(session_sizes.selectExpr(
  "percentile_approx(count, 0.5) as p50",
  "percentile_approx(count, 0.9) as p90",
  "percentile_approx(count, 0.99) as p99",
  "max(count) as max_count"
))


# COMMAND ----------

display(train_df.select("sensor_type").distinct().orderBy("sensor_type"))

# COMMAND ----------

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

sess_sensor = (
  train_df.groupBy("session_id", "sensor_type")
          .agg(*agg_exprs)
)

# COMMAND ----------

metric_cols = [c for c in sess_sensor.columns if c not in ("session_id", "sensor_type")]

sess_wide = (
  sess_sensor.groupBy("session_id")
             .pivot("sensor_type")
             .agg(*[F.first(c) for c in metric_cols])
)


# COMMAND ----------

print("sessions in train:", train_df.select("session_id").distinct().count())
print("rows in sess_wide:", sess_wide.count())

# COMMAND ----------

def clean_col(colname: str) -> str:
    import re
    m = re.match(r"^(\d+)_first\((.+)\)$", colname)
    if m:
        sensor, metric = m.group(1), m.group(2)
        return f"sensor_{sensor}_{metric}"
    return colname

for c in sess_wide.columns:
    sess_wide = sess_wide.withColumnRenamed(c, clean_col(c))

labels = train_df.select("session_id", "user_id").distinct()

train_features = sess_wide.join(labels, on="session_id", how="left")
display(train_features.limit(5))

# COMMAND ----------

sess_sensor_test = (
  test_df.groupBy("session_id", "sensor_type")
         .agg(*agg_exprs)
)

metric_cols_test = [c for c in sess_sensor_test.columns if c not in ("session_id", "sensor_type")]

test_wide = (
  sess_sensor_test.groupBy("session_id")
                  .pivot("sensor_type")
                  .agg(*[F.first(c) for c in metric_cols_test])
)

for c in test_wide.columns:
    test_wide = test_wide.withColumnRenamed(c, clean_col(c))

test_features = test_wide

train_cols = set(train_features.columns) - {"user_id"}
test_cols  = set(test_features.columns)

all_feature_cols = sorted((train_cols | test_cols) - {"session_id"})

# add missing columns as null, then fill with 0
for c in all_feature_cols:
    if c not in train_features.columns:
        train_features = train_features.withColumn(c, F.lit(None).cast("double"))
    if c not in test_features.columns:
        test_features = test_features.withColumn(c, F.lit(None).cast("double"))

train_features = train_features.select(["session_id"] + all_feature_cols + ["user_id"]).fillna(0)
test_features  = test_features.select(["session_id"] + all_feature_cols).fillna(0)

# COMMAND ----------

user_dist = (train_df.groupBy("user_id").count().orderBy(F.desc("count")))
ud = user_dist.toPandas()
plt.figure()
plt.bar(range(len(ud)), ud["count"])
plt.xticks(range(len(ud)), ud["user_id"], rotation=90)
plt.ylabel("sessions/events count") 
plt.show()

# COMMAND ----------

pdf_train = train_features.toPandas()
pdf_test  = test_features.toPandas()

X = pdf_train.drop(columns=["session_id", "user_id"])
y = pdf_train["user_id"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = Pipeline(steps=[
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("lr", LogisticRegression(
        max_iter=2000,
        C=1.0,              # inverse of regularization strength
        multi_class="auto",
        n_jobs=-1
    ))
])

clf.fit(X_train, y_train)
pred = clf.predict(X_valid)
print("Validation accuracy:", accuracy_score(y_valid, pred))
print("Validation F1 Score:", f1_score(y_valid, pred, average='macro'))


# COMMAND ----------

# Train on full training set
clf.fit(X, y)

X_test = pdf_test.drop(columns=["session_id"])
test_pred = clf.predict(X_test)

submission = pd.DataFrame({
    "session_id": pdf_test["session_id"],
    "user_id": test_pred
})

display(submission.head())
print("submission rows:", len(submission))


# COMMAND ----------

le = LabelEncoder()

# fit encoder on ALL labels
y_enc = le.fit_transform(y)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
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
    random_state=42
)

xgb.fit(X_train, y_train)
pred_xgb = xgb.predict(X_valid)
print("XGB validation accuracy:", accuracy_score(y_valid, pred_xgb))
print("XGB F1 Score:", f1_score(y_valid, pred_xgb, average='macro'))


# COMMAND ----------

seeds = [0, 1, 2, 3, 4]

dummy_mf_scores, dummy_strat_scores = [], []
lr_scores, svm_scores, rf_scores, xgb_scores = [], [], [], []

# XGBoost needs encoded labels
le = LabelEncoder()
y_enc_all = le.fit_transform(y)

for seed in seeds:
    # ----- Models that can use string labels -----
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    dummy_mf = DummyClassifier(strategy="most_frequent", random_state=seed)
    dummy_mf.fit(X_tr, y_tr)
    dummy_mf_scores.append(f1_score(y_va, dummy_mf.predict(X_va), average="macro"))

    dummy_strat = DummyClassifier(strategy="stratified", random_state=seed)
    dummy_strat.fit(X_tr, y_tr)
    dummy_strat_scores.append(f1_score(y_va, dummy_strat.predict(X_va), average="macro"))

    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=3000, C=1.0, n_jobs=-1))
    ])
    lr.fit(X_tr, y_tr)
    lr_scores.append(f1_score(y_va, lr.predict(X_va), average="macro"))

    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(C=1.0))
    ])
    svm.fit(X_tr, y_tr)
    svm_scores.append(f1_score(y_va, svm.predict(X_va), average="macro"))

    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=seed,
        class_weight="balanced_subsample"
    )
    rf.fit(X_tr, y_tr)
    rf_scores.append(f1_score(y_va, rf.predict(X_va), average="macro"))

    # ----- XGB needs integer labels -----
    X_tr, X_va, y_tr, y_va = train_test_split(
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
    )
    xgb.fit(X_tr, y_tr)
    xgb_scores.append(f1_score(y_va, xgb.predict(X_va), average="macro"))

print("Macro-F1 means")
print("Dummy most_frequent:", dummy_mf_scores, "mean:", np.mean(dummy_mf_scores))
print("Dummy stratified   :", dummy_strat_scores, "mean:", np.mean(dummy_strat_scores))
print("LR                :", lr_scores, "mean:", np.mean(lr_scores))
print("SVM               :", svm_scores, "mean:", np.mean(svm_scores))
print("RF                :", rf_scores, "mean:", np.mean(rf_scores))
print("XGB               :", xgb_scores, "mean:", np.mean(xgb_scores))


# COMMAND ----------

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
        )
        rf.fit(X_tr, y_tr)
        scores.append(f1_score(y_va, rf.predict(X_va), average="macro"))

    results.append({
        "params": params,
        "mean": float(np.mean(scores)),
        "scores": scores
    })

results_sorted = sorted(results, key=lambda d: d["mean"], reverse=True)

for r in results_sorted[:10]:
    print(r["params"], "mean=", r["mean"], "scores=", r["scores"])


# COMMAND ----------

best_params = {'n_estimators': 300, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1}

rf_final = RandomForestClassifier(
    random_state=42,
    class_weight="balanced_subsample",
    **best_params
)

rf_final.fit(X, y)

X_test = pdf_test.drop(columns=["session_id"])
test_pred = rf_final.predict(X_test)

submission_rf = pd.DataFrame({
    "session_id": pdf_test["session_id"],
    "user_id": test_pred
})
submission_rf.to_csv("submission.csv", index=False)
print("Saved submission.csv rows:", len(submission_rf))


# COMMAND ----------

submission_final.tail(20)
