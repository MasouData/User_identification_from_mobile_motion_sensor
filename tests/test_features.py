import pytest
from pyspark.sql import functions as F

from src.data.loading import schema_train

from src.features.session_features import (
    build_session_features,
    align_train_test_features,
)


def _make_sensor_row(
    uid,
    timestamp,
    sensor_type,
    field_0,
    field_1,
    field_2,
    session_id,
    user_id,
):
    """
    Create one synthetic raw sensor event
    compatible with schema_train.
    """

    return (
        uid,
        timestamp,
        sensor_type,

        field_0,
        field_1,
        field_2,

        0.0,
        0.0,
        0.0,
        0.0,
        0.0,

        session_id,
        user_id,
    )


def _create_training_events(spark):
    """
    Small deterministic sensor dataset.

    session_1:
        sensor 1 -> field_0 values 1 and 3
        sensor 2 -> field_0 values 10 and 14

    session_2:
        sensor 1 -> field_0 values 2 and 4
    """

    rows = [
        _make_sensor_row(
            "event_1",
            1000,
            "1",
            1.0,
            2.0,
            3.0,
            "session_1",
            "user_1",
        ),

        _make_sensor_row(
            "event_2",
            2000,
            "1",
            3.0,
            4.0,
            5.0,
            "session_1",
            "user_1",
        ),

        _make_sensor_row(
            "event_3",
            1000,
            "2",
            10.0,
            11.0,
            12.0,
            "session_1",
            "user_1",
        ),

        _make_sensor_row(
            "event_4",
            2000,
            "2",
            14.0,
            15.0,
            16.0,
            "session_1",
            "user_1",
        ),

        _make_sensor_row(
            "event_5",
            1000,
            "1",
            2.0,
            3.0,
            4.0,
            "session_2",
            "user_2",
        ),

        _make_sensor_row(
            "event_6",
            2000,
            "1",
            4.0,
            5.0,
            6.0,
            "session_2",
            "user_2",
        ),
    ]

    return spark.createDataFrame(
        rows,
        schema=schema_train,
    )


def test_build_session_features(spark):
    """
    Raw events should become one row per session,
    and aggregation should produce the expected mean.
    """

    events_df = _create_training_events(
        spark
    )

    features = build_session_features(
        events_df,
        has_label=True,
    )

    # Two sessions should produce two feature rows.
    assert features.count() == 2

    assert "user_id" in features.columns

    assert (
        "sensor_1_field_0_mean"
        in features.columns
    )

    # session_1, sensor 1 has field_0:
    #
    # 1.0 and 3.0
    #
    # Mean should therefore be 2.0.
    row = (
        features
        .filter(
            F.col("session_id")
            == "session_1"
        )
        .select(
            "sensor_1_field_0_mean"
        )
        .first()
    )

    assert (
        row["sensor_1_field_0_mean"]
        == pytest.approx(2.0)
    )


def test_align_train_test_features(spark):
    """
    Train/test feature tables should end up with
    exactly the same ML columns.

    If a sensor exists only in training,
    its missing test features should be filled with 0.
    """

    train_events = (
        _create_training_events(
            spark
        )
    )

    train_features = (
        build_session_features(
            train_events,
            has_label=True,
        )
    )

    # Test data contains only session_2.
    #
    # session_2 contains sensor 1,
    # but does not contain sensor 2.
    test_events = (
        train_events
        .filter(
            F.col("session_id")
            == "session_2"
        )
        .drop("user_id")
    )

    test_features = (
        build_session_features(
            test_events,
            has_label=False,
        )
    )

    (
        aligned_train,
        aligned_test,
        feature_columns,
    ) = align_train_test_features(
        train_features,
        test_features,
    )

    # Training has:
    # session_id + features + user_id
    assert (
        len(aligned_train.columns)
        == len(feature_columns) + 2
    )

    # Test has:
    # session_id + features
    assert (
        len(aligned_test.columns)
        == len(feature_columns) + 1
    )

    assert (
        "sensor_2_field_0_mean"
        in aligned_test.columns
    )

    test_row = (
        aligned_test
        .filter(
            F.col("session_id")
            == "session_2"
        )
        .first()
    )

    # Sensor 2 was absent from this test session,
    # so the aligned feature should be zero.
    assert (
        test_row["sensor_2_field_0_mean"]
        == pytest.approx(0.0)
    )