import pytest

from src.data.validation import (
    get_null_counts,
    find_mixed_user_sessions,
    get_imbalance_ratio,
)


def test_null_counts(spark):
    """
    Null-count validation should correctly detect
    missing values.
    """

    df = spark.createDataFrame(
        [
            ("session_1", "user_1", "1"),
            ("session_2", None, "2"),
            ("session_3", "user_2", None),
        ],
        [
            "session_id",
            "user_id",
            "sensor_type",
        ],
    )

    result = (
        get_null_counts(df)
        .first()
        .asDict()
    )

    assert result["session_id"] == 0
    assert result["user_id"] == 1
    assert result["sensor_type"] == 1


def test_find_mixed_user_sessions(spark):
    """
    A session belonging to two users should be detected.
    """

    df = spark.createDataFrame(
        [
            ("session_1", "user_1"),
            ("session_1", "user_1"),

            ("session_2", "user_2"),
            ("session_2", "user_3"),
        ],
        [
            "session_id",
            "user_id",
        ],
    )

    mixed_sessions = (
        find_mixed_user_sessions(df)
        .collect()
    )

    assert len(mixed_sessions) == 1

    assert (
        mixed_sessions[0]["session_id"]
        == "session_2"
    )

    assert (
        mixed_sessions[0]["n_users"]
        == 2
    )


def test_imbalance_ratio(spark):
    """
    If user_1 has one session and user_2 has two,
    the imbalance ratio should equal 2.
    """

    df = spark.createDataFrame(
        [
            ("session_1", "user_1"),
            ("session_2", "user_2"),
            ("session_3", "user_2"),
        ],
        [
            "session_id",
            "user_id",
        ],
    )

    ratio = get_imbalance_ratio(df)

    assert ratio == pytest.approx(2.0)