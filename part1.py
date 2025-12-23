# Databricks notebook source
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
