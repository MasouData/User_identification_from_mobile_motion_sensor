# Databricks notebook source
# MAGIC %pip install xgboost

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

from pyspark.sql.types import *

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

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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


# COMMAND ----------

# Train on full training set
clf.fit(X, y)

# Predict on test
X_test = pdf_test.drop(columns=["session_id"])
test_pred = clf.predict(X_test)

submission = pd.DataFrame({
    "session_id": pdf_test["session_id"],
    "user_id": test_pred
})

display(submission.head())
print("submission rows:", len(submission))


# COMMAND ----------

submission.to_csv("submission.csv", index=False)

# COMMAND ----------

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

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


# COMMAND ----------

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

seeds = [0, 1, 2, 3, 4]
lr_scores, xgb_scores = [], []

# prepare encoded labels for XGB
le = LabelEncoder()
y_enc = le.fit_transform(y)

for seed in seeds:
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    lr = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=2000, n_jobs=-1))])
    lr.fit(X_tr, y_tr)
    lr_scores.append(accuracy_score(y_va, lr.predict(X_va)))

    X_tr, X_va, y_tr, y_va = train_test_split(X, y_enc, test_size=0.2, random_state=seed, stratify=y_enc)
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
    xgb_scores.append(accuracy_score(y_va, xgb.predict(X_va)))

print("LR scores:", lr_scores, "mean:", np.mean(lr_scores))
print("XGB scores:", xgb_scores, "mean:", np.mean(xgb_scores))


# COMMAND ----------

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

Cs = [0.1, 1.0, 10.0]
seeds = [0, 1, 2, 3, 4]

results = {}

for C in Cs:
    scores = []
    for seed in seeds:
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        lr = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=3000, C=C, n_jobs=-1))
        ])
        lr.fit(X_tr, y_tr)
        scores.append(accuracy_score(y_va, lr.predict(X_va)))
    results[C] = (scores, float(np.mean(scores)))

for C, (scores, mean_score) in results.items():
    print(f"C={C} scores={scores} mean={mean_score}")


# COMMAND ----------

best_C = 1.0

final_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=3000, C=best_C, n_jobs=-1))
])

final_lr.fit(X, y)

X_test = pdf_test.drop(columns=["session_id"])
test_pred = final_lr.predict(X_test)

submission_final = pd.DataFrame({
    "session_id": pdf_test["session_id"],
    "user_id": test_pred
})

submission_final.to_csv("submission.csv", index=False)
print("Saved submission.csv with rows:", len(submission_final))

