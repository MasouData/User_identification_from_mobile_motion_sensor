# Databricks notebook source
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

