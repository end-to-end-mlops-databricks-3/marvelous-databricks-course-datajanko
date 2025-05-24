"""Unit tests for BasicModel."""

import mlflow
import pandas as pd
from conftest import CATALOG_DIR, TRACKING_URI
from loguru import logger
from mlflow.entities.model_registry.registered_model import RegisteredModel
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession

from mlops_course.config import ProjectConfig, Tags
from mlops_course.models.basic_model import BasicModel

mlflow.set_tracking_uri(TRACKING_URI)


def test_custom_model_init(model_config: ProjectConfig, tags: Tags, spark_session: SparkSession) -> None:
    """Test the initialization of BasicModel.

    This function creates a BasicModel instance and asserts that its attributes are of the correct types.

    :param config: Configuration for the project
    :param tags: Tags associated with the model
    :param spark_session: Spark session object
    """
    model = BasicModel(config=model_config, tags=tags, spark=spark_session)
    assert isinstance(model, BasicModel)
    assert isinstance(model.config, ProjectConfig)
    assert isinstance(model.tags, dict)
    assert isinstance(model.spark, SparkSession)


def test_load_data_validate(mock_basic_model: BasicModel) -> None:
    """Verify correct feature/target splits in training and test data.

    :param mock_basic_model: Mocked BasicModel instance for testing.
    """
    # Removed the assignment test. Since we do not read from spark, there was no real value in here due to lack of transformation

    train_data = pd.read_csv((CATALOG_DIR / "train_set_for_mock.csv").as_posix()).assign(
        date=lambda df: pd.to_datetime(df["date"])
    )
    test_data = pd.read_csv((CATALOG_DIR / "test_set_for_mock.csv").as_posix()).assign(
        date=lambda df: pd.to_datetime(df["date"])
    )

    # Execute
    mock_basic_model.load_data()

    # Verify feature/target splits
    expected_features = mock_basic_model.features
    pd.testing.assert_frame_equal(mock_basic_model.X_train, train_data.loc[2:5, expected_features])
    pd.testing.assert_series_equal(mock_basic_model.y_train, train_data.loc[2:5, mock_basic_model.target])
    pd.testing.assert_frame_equal(mock_basic_model.X_valid, train_data.loc[0:1, expected_features])
    pd.testing.assert_series_equal(mock_basic_model.y_valid, train_data.loc[0:1, mock_basic_model.target])
    pd.testing.assert_frame_equal(mock_basic_model.X_test, test_data[expected_features])
    pd.testing.assert_series_equal(mock_basic_model.y_test, test_data[mock_basic_model.target])


def test_train(mock_basic_model: BasicModel) -> None:
    """Test that train method configures pipeline with correct feature handling.

    Validates feature count matches configuration and feature names align with
    numerical/categorical features defined in model config.

    :param mock_basic_model: Mocked BasicModel instance for testing
    """
    mock_basic_model.load_data()
    mock_basic_model.create_pipeline()
    mock_basic_model.train()
    expected_feature_names = mock_basic_model.features

    assert mock_basic_model.pipeline.n_features_in_ == len(expected_feature_names)
    assert sorted(expected_feature_names) == sorted(mock_basic_model.pipeline.feature_names_in_)


def test_log_model_with_PandasDataset(mock_basic_model: BasicModel) -> None:
    """Test model logging with PandasDataset validation.

    Verifies that the model's pipeline captures correct feature dimensions and names,
    then checks proper dataset type handling during model logging.

    :param mock_basic_model: Mocked BasicModel instance for testing
    """
    mock_basic_model.load_data()
    mock_basic_model.create_pipeline()
    mock_basic_model.train()
    expected_feature_names = mock_basic_model.features

    assert mock_basic_model.pipeline.n_features_in_ == len(expected_feature_names)
    assert sorted(expected_feature_names) == sorted(mock_basic_model.pipeline.feature_names_in_)

    mock_basic_model.log_model(dataset_type="PandasDataset")

    # Split the following part
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(mock_basic_model.experiment_name)
    assert experiment.name == mock_basic_model.experiment_name

    experiment_id = experiment.experiment_id
    assert experiment_id

    runs = client.search_runs(experiment_id, order_by=["start_time desc"], max_results=1)
    assert len(runs) == 1
    latest_run = runs[0]

    model_uri = f"runs:/{latest_run.info.run_id}/model"
    logger.info(f"{model_uri= }")

    assert model_uri


def test_register_model(mock_basic_model: BasicModel) -> None:
    """Test the registration of a custom MLflow model.

    This function performs several operations on the mock custom model, including loading data,
    preparing features, training, and logging the model. It then registers the model and verifies
    its existence in the MLflow model registry.

    :param mock_basic_model: A mocked instance of the BasicModel class.
    """
    mock_basic_model.load_data()
    mock_basic_model.create_pipeline()
    mock_basic_model.train()
    mock_basic_model.log_model(dataset_type="PandasDataset")

    mock_basic_model.register_model()

    client = MlflowClient()
    model_name = f"{mock_basic_model.catalog_name}.{mock_basic_model.schema_name}.cs_go_model_basic"

    try:
        model = client.get_registered_model(model_name)
        logger.info(f"Model '{model_name}' is registered.")
        logger.info(f"Latest version: {model.latest_versions[-1].version}")
        logger.info(f"{model.name = }")
    except mlflow.exceptions.MlflowException as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e):
            logger.error(f"Model '{model_name}' is not registered.")
        else:
            raise e

    assert isinstance(model, RegisteredModel)
    alias, version = model.aliases.popitem()
    assert alias == "latest-model"


def test_retrieve_current_run_metadata(mock_basic_model: BasicModel) -> None:
    """Test retrieving the current run metadata from a mock custom model.

    This function verifies that the `retrieve_current_run_metadata` method
    of the `BasicModel` class returns metrics and parameters as dictionaries.

    :param mock_basic_model: A mocked instance of the BasicModel class.
    """
    mock_basic_model.load_data()
    mock_basic_model.create_pipeline()
    mock_basic_model.train()
    mock_basic_model.log_model(dataset_type="PandasDataset")

    metrics, params = mock_basic_model.retrieve_current_run_metadata()
    assert isinstance(metrics, dict)
    assert metrics
    assert isinstance(params, dict)
    assert params


def test_load_latest_model_and_predict(mock_basic_model: BasicModel) -> None:
    """Test the process of loading the latest model and making predictions.

    This function performs the following steps:
    - Loads data using the provided custom model.
    - Prepares features and trains the model.
    - Logs and registers the trained model.
    - Extracts input data from the test set and makes predictions using the latest model.

    :param mock_basic_model: Instance of a custom machine learning model with methods for data
                              loading, feature preparation, training, logging, and prediction.
    """
    mock_basic_model.load_data()
    mock_basic_model.create_pipeline()
    mock_basic_model.train()
    mock_basic_model.log_model(dataset_type="PandasDataset")
    mock_basic_model.register_model()

    input_data = mock_basic_model.test_set_pandas.drop(columns=[mock_basic_model.target])
    input_data = input_data.where(input_data.notna(), None)  # noqa

    for row in input_data.itertuples(index=False):
        row_df = pd.DataFrame([row._asdict()])
        print(row_df.to_dict(orient="split"))
        predictions = mock_basic_model.load_latest_model_and_predict(input_data=row_df)

        assert len(predictions) == 1
