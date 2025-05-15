"""Dataloader fixture."""

from pathlib import Path

import pandas as pd
import pytest
from loguru import logger
from numpy import dtype
from pandas import CategoricalDtype
from pyspark.sql import SparkSession

from mlops_course import PROJECT_DIR
from mlops_course.config import ProjectConfig, Tags
from tests.unit_tests.spark_config import spark_config


@pytest.fixture(scope="session")
def spark_session() -> SparkSession:
    """Create and return a SparkSession for testing.

    This fixture creates a SparkSession with the specified configuration and returns it for use in tests.
    """
    # One way
    # spark = SparkSession.builder.getOrCreate()  # noqa
    # Alternative way - better
    spark = (
        SparkSession.builder.master(spark_config.master)
        .appName(spark_config.app_name)
        .config("spark.executor.cores", spark_config.spark_executor_cores)
        .config("spark.executor.instances", spark_config.spark_executor_instances)
        .config("spark.sql.shuffle.partitions", spark_config.spark_sql_shuffle_partitions)
        .config("spark.driver.bindAddress", spark_config.spark_driver_bindAddress)
        .getOrCreate()
    )

    yield spark  # noqa
    spark.stop()


@pytest.fixture(scope="session")
def config() -> ProjectConfig:
    """Load and return the project configuration.

    This fixture reads the project configuration from a YAML file and returns a ProjectConfig object.

    :return: The loaded project configuration
    """
    config_file_path = (PROJECT_DIR / "project_config.yml").resolve()
    logger.info(f"Current config file path: {config_file_path.as_posix()}")
    config = ProjectConfig.from_yaml(config_file_path.as_posix())
    return config


@pytest.fixture
def raw_data_file_path() -> Path:
    """Provide the project config as a fixture."""
    return PROJECT_DIR / "tests" / "test_data" / "raw_samples.csv"


@pytest.fixture(scope="function")
def typed_data_sample() -> pd.DataFrame:
    """Provide a typed dataframe from result samples.

    This fixture reads a CSV file using pandas

    :return: A sampled Pandas DataFrame containing some sample of typed dat
    """
    file_path = PROJECT_DIR / "tests" / "test_data" / "processed_samples.csv"
    dtypes = {
        "team_1": CategoricalDtype(
            categories=[
                "AGO",
                "AVANT",
                "Astralis",
                "Bad News Bears",
                "Chaos",
                "Dignitas",
                "Divine",
                "EHOME",
                "FURIA",
                "FunPlus Phoenix",
                "GamerLegion",
                "Gen.G",
                "HAVU",
                "HellRaisers",
                "I sleep",
                "Illuminar",
                "In The Lab",
                "Infinity",
                "MAD Lions",
                "New England Whalers",
                "ORDER",
                "PC419",
                "Reapers",
                "Recon 5",
                "Station7",
                "Syman",
                "TeamOne",
                "Thunder Logic",
                "Triumph",
                "Under 21",
                "fnatic",
            ],
            ordered=False,
        ),
        "team_2": CategoricalDtype(
            categories=[
                "BIG",
                "Big Frames",
                "Chiefs",
                "Cloud9",
                "Complexity",
                "Divine",
                "Envy",
                "Evil Geniuses",
                "FURY",
                "Formidable",
                "Illuminar",
                "Infinity",
                "JiJieHao",
                "KOVA",
                "Mythic",
                "Na`Vi Junior",
                "Neverest",
                "Nordavind",
                "Oceanus",
                "Orgless",
                "Rugratz",
                "Skyfire",
                "Station7",
                "TeamOne",
                "Under 21",
                "Unicorns of Love",
                "fbg",
                "fnatic",
            ],
            ordered=False,
        ),
        "map_name": CategoricalDtype(
            categories=["Dust2", "Inferno", "Mirage", "Nuke", "Overpass", "Train", "Vertigo"], ordered=False
        ),
        "result_1": dtype("int64"),
        "result_2": dtype("int64"),
        "map_winner": dtype("int64"),
        "starting_ct": dtype("int64"),
        "ct_1": dtype("int64"),
        "t_2": dtype("int64"),
        "t_1": dtype("int64"),
        "ct_2": dtype("int64"),
        "rank_1": dtype("int64"),
        "rank_2": dtype("int64"),
    }
    sample = pd.read_csv(file_path.as_posix(), dtype=dtypes, converters={"date": pd.to_datetime})

    return sample


@pytest.fixture(scope="session")
def tags() -> Tags:
    """Create and return a Tags instance for the test session.

    This fixture provides a Tags object with predefined values for git_sha, branch, and job_run_id.
    """
    return Tags(git_sha="wxyz", branch="test", job_run_id="9")
