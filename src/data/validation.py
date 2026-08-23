from pyspark.sql import functions as F


def get_null_counts(df):
    """
    Count null values in every column.

    Returns
    -------
    pyspark.sql.DataFrame
        One-row DataFrame containing the number of nulls per column.
    """
    return df.select([
        F.sum(F.col(c).isNull().cast("int")).alias(c)
        for c in df.columns
    ])


def find_mixed_user_sessions(df):
    """
    Find sessions associated with more than one user.

    This is a data-quality check for the training dataset:
    one session_id should belong to exactly one user_id.

    Returns
    -------
    pyspark.sql.DataFrame
        Sessions where more than one distinct user_id is present.
    """
    return (
        df.groupBy("session_id")
        .agg(F.countDistinct("user_id").alias("n_users"))
        .filter(F.col("n_users") > 1)
    )


def get_session_size_stats(df):
    """
    Calculate event-count statistics per session.

    Returns p50, p90, p99 and maximum number of events
    observed in a session.
    """
    session_sizes = (
        df.groupBy("session_id")
        .count()
    )

    return session_sizes.selectExpr(
        "percentile_approx(count, 0.50) AS p50",
        "percentile_approx(count, 0.90) AS p90",
        "percentile_approx(count, 0.99) AS p99",
        "max(count) AS max_count"
    )


def get_sensor_types(df):
    """
    Return the distinct sensor types available in the dataset.
    """
    return (
        df.select("sensor_type")
        .distinct()
        .orderBy("sensor_type")
    )


def get_user_session_counts(df):
    """
    Count the number of distinct sessions for each user.

    Used to inspect class balance in the training dataset.
    """
    return (
        df.groupBy("user_id")
        .agg(
            F.countDistinct("session_id").alias("n_sessions")
        )
        .orderBy(F.desc("n_sessions"))
    )


def get_imbalance_ratio(df):
    """
    Calculate:

        largest number of sessions for a user
        -------------------------------------
        smallest number of sessions for a user

    A value close to 1 indicates a balanced dataset.
    """
    user_counts = get_user_session_counts(df)

    summary = (
        user_counts
        .agg(
            F.max("n_sessions").alias("max_sessions"),
            F.min("n_sessions").alias("min_sessions")
        )
        .first()
    )

    if summary["min_sessions"] == 0:
        return None

    return (
        summary["max_sessions"]
        / summary["min_sessions"]
    )