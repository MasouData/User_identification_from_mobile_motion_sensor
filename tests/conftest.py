import sys
from pathlib import Path

import pytest
from pyspark.sql import SparkSession
# ---------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )


@pytest.fixture(scope="session")
def spark():
    """
    Provide one Spark session for all Spark-based tests.

    In Databricks, reuse the active Spark session.
    When running locally, create a small local session.
    """

    session = SparkSession.getActiveSession()

    if session is None:
        session = (
            SparkSession.builder
            .master("local[1]")
            .appName("motion-user-identification-tests")
            .getOrCreate()
        )

    return session