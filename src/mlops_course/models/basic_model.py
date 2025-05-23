"""Basic model implementation.

infer_signature (from mlflow.models) → Captures input-output schema for model tracking.

num_features → List of numerical feature names.
cat_features → List of categorical feature names.
target → The column to predict.
parameters → Hyperparameters for LightGBM.
catalog_name, schema_name → Database schema names for Databricks tables.
"""

from typing import Any

import mlflow
import mlflow.data
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from loguru import logger
from mlflow import MlflowClient
from mlflow.data.dataset_source import DatasetSource
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, log_loss
from sklearn.pipeline import Pipeline

from mlops_course.config import ProjectConfig, Tags
from mlops_course.transformers import CategoryTransformer


class BasicModel:
    """A basic model class for house price prediction using LightGBM.

    This class handles data loading, feature preparation, model training, and MLflow logging.
    """

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize the model with project configuration.

        :param config: Project configuration object
        :param tags: Tags object
        :param spark: SparkSession object
        """
        self.config = config
        self.spark = spark

        # Extract settings from the config
        self.features = self.config.selection.features
        self.cat_features = self.config.selection.categories
        self.target = self.config.selection.target
        self.parameters = self.config.parameters
        self.fit_parameters = self.config.fit_parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.experiment_name = self.config.experiment_name_basic
        self.model_name = f"{self.catalog_name}.{self.schema_name}.cs_go_model_basic"
        self.validation_start = self.config.validation_start_day
        self.tags = tags.dict()

    def load_data(self) -> None:
        """Load training and testing data from Delta tables.

        Splits data into features (X_train, X_test) and target (y_train, y_test).
        """
        logger.info("🔄 Loading data from Databricks tables...")
        self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        self.train_set = self.train_set_spark.toPandas()
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set").toPandas()
        self.data_version = "0"  # describe history -> retrieve

        train = self.train_set.query("date < @self.validation_start")
        valid = self.train_set.query("date >= @self.validation_start")
        self.X_train = train[self.features]
        self.y_train = train[self.target]
        self.X_valid = valid[self.features]
        self.y_valid = valid[self.target]
        self.X_test = self.test_set[self.features]
        self.y_test = self.test_set[self.target]

        logger.info("✅ Data successfully loaded.")

    def create_pipeline(self) -> None:
        """Provide Pipeline for preprocessing and training.

        Uses a custom CategoryTransformer to fit catgeories seen during training
        """
        category_transformers = [
            (col, CategoryTransformer().set_output(transform="pandas"), col) for col in self.cat_features
        ]
        self.pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    ColumnTransformer(
                        transformers=category_transformers,
                        remainder="passthrough",
                        verbose_feature_names_out=False,
                    ).set_output(transform="pandas"),
                ),
                ("lgbm", LGBMClassifier(**self.parameters)),
            ]
        )

        logger.info("✅ Preprocessing pipeline defined.")

    def train(self) -> None:
        """Train the model."""
        logger.info("🚀 Starting training...")

        # Unfortunately metadata routing doesn't work yet. E.g. pipelines transform_input is available from sklearn 1.6
        # Hence we decompose the pipeline
        preproc = self.pipeline.named_steps["preprocessor"]
        model = self.pipeline.named_steps["lgbm"]
        X_train = preproc.fit_transform(self.X_train)
        X_valid = preproc.transform(self.X_valid)
        self.pipeline.named_steps["pre_processor"] = preproc
        callbacks: list[Any] = [log_evaluation(10)]
        if "early_stopping" in self.fit_parameters:
            callbacks.append(early_stopping(int(self.fit_parameters["early_stopping"])))
        model.fit(X_train, self.y_train, eval_set=[(X_valid, self.y_valid)], callbacks=callbacks)
        self.pipeline.named_steps["lgbm"] = model
        logger.info("🚀 Training completed!")

    def log_model(self) -> None:
        """Log the model using MLflow."""
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(tags=self.tags) as run:
            model = self.pipeline.named_steps["lgbm"]
            mlflow.log_dict({"best_valid_loss": model.best_score_}, "best_score.json")
            mlflow.log_metric("best_iteration", model.best_iteration_)
            self.run_id = run.info.run_id

            y_pred_proba = self.pipeline.predict_proba(self.X_test)
            y_pred = self.pipeline.predict(self.X_test)

            # Evaluate metrics
            ll = log_loss(self.y_test, y_pred_proba)
            report = classification_report(self.y_test, y_pred)

            logger.info(f"📊 Test log loss: {ll}")

            # Log parameters and metrics
            mlflow.log_param("model_type", "LightGBM with preprocessing")
            mlflow.log_metric("test_log_loss", ll)  # noqa
            mlflow.log_dict({"report": report}, "report.json")

            # Log the model
            data_version = "0"
            signature = infer_signature(model_input=self.X_train, model_output=y_pred_proba)
            dataset = mlflow.data.from_spark(
                self.train_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.train_set",
                version=data_version,
            )
            mlflow.log_input(dataset, context="training")
            mlflow.sklearn.log_model(
                sk_model=self.pipeline,
                artifact_path="lightgbm-pipeline-model",
                signature=signature,
                pyfunc_predict_fn="predict_proba",
            )
            print(y_pred_proba)

    def register_model(self) -> None:
        """Register model in Unity Catalog."""
        logger.info("🔄 Registering the model in UC...")
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/lightgbm-pipeline-model",
            name=self.model_name,
            tags=self.tags,
        )
        logger.info(f"✅ Model registered as version {registered_model.version}.")

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=self.model_name,
            alias="latest-model",
            version=latest_version,
        )

    def retrieve_current_run_dataset(self) -> DatasetSource:
        """Retrieve MLflow run dataset.

        :return: Loaded dataset source
        """
        run = mlflow.get_run(self.run_id)
        dataset_info = run.inputs.dataset_inputs[0].dataset
        dataset_source = mlflow.data.get_source(dataset_info)
        logger.info("✅ Dataset source loaded.")
        return dataset_source.load()

    def retrieve_current_run_metadata(self) -> tuple[dict, dict]:
        """Retrieve MLflow run metadata.

        :return: Tuple containing metrics and parameters dictionaries
        """
        run = mlflow.get_run(self.run_id)
        metrics = run.data.to_dictionary()["metrics"]
        params = run.data.to_dictionary()["params"]
        logger.info("✅ Dataset metadata loaded.")
        return metrics, params

    def load_latest_model_and_predict(self, input_data: pd.DataFrame) -> np.ndarray:
        """Load the latest model from MLflow (alias=latest-model) and make predictions.

        Alias latest is not allowed -> we use latest-model instead as an alternative.

        :param input_data: Pandas DataFrame containing input features for prediction.
        :return: Pandas DataFrame with predictions.
        """
        logger.info("🔄 Loading model from MLflow alias 'production'...")

        model_uri = f"models:/{self.model_name}@latest-model"
        model = mlflow.sklearn.load_model(model_uri)

        logger.info("✅ Model successfully loaded.")

        # Make predictions
        predictions = model.predict(input_data)

        # Return predictions as a DataFrame
        return predictions
