from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    LongType,
    DoubleType,
)


schema_train = StructType([
    StructField("uid", StringType()),
    StructField("timestamp", LongType()),
    StructField("sensor_type", StringType()),
    *[StructField(f"field_{i}", DoubleType()) for i in range(8)],
    StructField("session_id", StringType()),
    StructField("user_id", StringType()),
])

schema_test = StructType([
    field for field in schema_train.fields
    if field.name != "user_id"
])


def load_train_data(spark, path):
    return (
        spark.read
        .option("header", True)
        .schema(schema_train)
        .csv(path)
    )


def load_test_data(spark, path):
    return (
        spark.read
        .option("header", True)
        .schema(schema_test)
        .csv(path)
    )