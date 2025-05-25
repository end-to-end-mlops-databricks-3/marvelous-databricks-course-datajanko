"""Model fixture."""

import shutil
from unittest.mock import MagicMock

import pandas as pd
import pytest
from conftest import CATALOG_DIR, MLRUNS_DIR
from loguru import logger
from pyspark.sql import SparkSession

from mlops_course import PROJECT_DIR
from mlops_course.config import ProjectConfig, Tags
from mlops_course.models.basic_model import BasicModel


@pytest.fixture(scope="session")
def model_config() -> ProjectConfig:
    """Load and return the project configuration.

    This fixture reads the project configuration from a YAML file and returns a ProjectConfig object.

    :return: The loaded project configuration
    """
    config_file_path = (PROJECT_DIR / "tests" / "test_data" / "test_config_for_basic_model.yml").resolve()
    logger.info(f"Current config file path: {config_file_path.as_posix()}")
    config = ProjectConfig.from_yaml(config_file_path.as_posix())
    return config


@pytest.fixture(scope="session", autouse=True)
def create_mlruns_directory() -> None:
    """Create or recreate the MLFlow tracking directory.

    This fixture ensures that the MLFlow tracking directory is clean and ready for use
    before each test session.
    """
    if MLRUNS_DIR.exists():
        shutil.rmtree(MLRUNS_DIR)
        MLRUNS_DIR.mkdir()
        logger.info(f"Created {MLRUNS_DIR} directory for MLFlow tracking")
    else:
        logger.info(f"MLFlow tracking directory {MLRUNS_DIR} does not exist")


@pytest.fixture(scope="function")
def mock_basic_model(model_config: ProjectConfig, tags: Tags, spark_session: SparkSession) -> BasicModel:
    """Fixture that provides a CustomModel instance with mocked Spark interactions.

    Initializes the model with test data and mocks Spark DataFrame conversions to pandas.

    :param config: Project configuration parameters
    :param tags: Tagging metadata for model tracking
    :param spark_session: Spark session instance for testing environment
    :return: Configured BasicModel instance with mocked Spark interactions
    """
    instance = BasicModel(
        config=model_config,
        tags=tags,
        spark=spark_session,
    )

    train_data = pd.read_csv((CATALOG_DIR / "train_set_for_mock.csv").as_posix()).assign(
        date=lambda df: pd.to_datetime(df["date"])
    )

    # Important Note: Replace NaN with None in Pandas
    # train_data = train_data.where(train_data.notna(), None)  # noqa

    test_data = pd.read_csv((CATALOG_DIR / "test_set_for_mock.csv").as_posix()).assign(
        date=lambda df: pd.to_datetime(df["date"])
    )

    test_data = test_data.where(test_data.notna(), None)  # noqa

    ## Mock Spark interactions
    # Mock Spark DataFrame with toPandas() method
    mock_spark_df_train = MagicMock()
    mock_spark_df_train.toPandas.return_value = train_data
    mock_spark_df_test = MagicMock()
    mock_spark_df_test.toPandas.return_value = test_data

    # Mock spark.table method
    mock_spark = MagicMock()
    mock_spark.table.side_effect = [mock_spark_df_train, mock_spark_df_test]
    instance.spark = mock_spark

    return instance
