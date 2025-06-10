"""Feature Lookup model implementation.

infer_signature (from mlflow.models) → Captures input-output schema for model tracking.

num_features → List of numerical feature names.
cat_features → List of categorical feature names.
target → The column to predict.
parameters → Hyperparameters for LightGBM.
catalog_name, schema_name → Database schema names for Databricks tables.
"""

# from mlops_course.config import ProjectConfig, Tags
from typing import Any, Literal, Self

import mlflow
import pandas as pd
from databricks.feature_engineering import FeatureEngineeringClient
from databricks.feature_store import FeatureLookup
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from loguru import logger
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pyspark.sql import DataFrame, SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, log_loss
from sklearn.pipeline import Pipeline

from mlops_course.config import ProjectConfig, Tags


class FeatureLookupModel:
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
        self.fe = FeatureEngineeringClient()

        # Extract settings from the config
        self.features = self.config.selection.features
        self.cat_features = self.config.selection.categories
        self.target = self.config.selection.target
        self.parameters = self.config.parameters
        self.fit_parameters = self.config.fit_parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.experiment_name_fe = self.config.experiment_name_fe
        self.model_name = f"{self.catalog_name}.{self.schema_name}.cs_go_model_feature_lookup"
        self.validation_start = self.config.validation_start_day
        self.tags = tags.dict()
        self.overall_feature_table_name = f"{config.catalog_name}.{config.schema_name}.overall_winning_shares_fe_v2"
        self.per_map_feature_table_name = f"{config.catalog_name}.{config.schema_name}.per_map_winning_shares_fe_v2"

    def load_data(self) -> None:
        """Load training and testing data from Delta tables.

        Splits data into features (X_train, X_test) and target (y_train, y_test).
        """
        logger.info("🔄 Loading data from Databricks tables...")
        self.training_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set")
        self.data_version = "0"  # describe history -> retrieve

        train_set = self.fe.create_training_set(
            df=self.training_set,
            label=self.target,
            feature_lookups=[
                FeatureLookup(
                    table_name=self.overall_feature_table_name,
                    lookup_key=["team_1"],
                    timestamp_lookup_key="date",
                    feature_names=["last_10_match_day_winshare", "last_10_days_match_count"],
                    rename_outputs={
                        "last_10_match_day_winshare": "team_1_last_10_match_day_winshare",
                        "last_10_days_match_count": "team_1_last_10_match_day_match_count",
                    },
                ),
                FeatureLookup(
                    table_name=self.per_map_feature_table_name,
                    lookup_key=["map_name", "team_1"],
                    timestamp_lookup_key="date",
                    feature_names=["last_10_match_day_win_share_for_map", "last_10_days_match_count_for_map"],
                    rename_outputs={
                        "last_10_match_day_win_share_for_map": "team_1_last_10_match_day_win_share_for_map",
                        "last_10_days_match_count_for_map": "team_1_last_10_match_day_match_count_for_map",
                    },
                ),
                FeatureLookup(
                    table_name=self.overall_feature_table_name,
                    lookup_key=["team_2"],
                    timestamp_lookup_key="date",
                    feature_names=["last_10_match_day_winshare", "last_10_days_match_count"],
                    rename_outputs={
                        "last_10_match_day_winshare": "team_2_last_10_match_day_winshare",
                        "last_10_days_match_count": "team_2_last_10_match_day_match_count",
                    },
                ),
                FeatureLookup(
                    table_name=self.per_map_feature_table_name,
                    lookup_key=["map_name", "team_2"],
                    timestamp_lookup_key="date",
                    feature_names=["last_10_match_day_win_share_for_map", "last_10_days_match_count_for_map"],
                    rename_outputs={
                        "last_10_match_day_win_share_for_map": "team_2_last_10_match_day_win_share_for_map",
                        "last_10_days_match_count_for_map": "team_2_last_10_match_day_match_count_for_map",
                    },
                ),
            ],  # , exclude_columns='date' we exclude this for now as we use pandas for train/validation split. If we conduct this on the spark level, we'd need to make the above training set a function and call it twice as this is only available once the model is registered
        )

        # Since we're using LGBM and it can deal with nans we are lazy and do not implement FeatureFunctions to impute non found lookups and also do not
        # add an imputer to the pipeline
        self.train_set = train_set
        train_df = train_set.load_df().toPandas()

        train = train_df.query("date < @self.validation_start")
        valid = train_df.query("date >= @self.validation_start")
        self.X_train = train.drop(["date", self.target], axis=1)
        self.y_train = train[self.target]
        self.X_valid = valid.drop(["date", self.target], axis=1)
        self.y_valid = valid[self.target]

        logger.info("✅ Data successfully loaded.")

    def create_pipeline(self) -> None:
        """Provide Pipeline for preprocessing and training.

        Uses a custom CategoryTransformer to fit catgeories seen during training
        """
        from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin

        class CategoryTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
            """Transformer for treating categorical variables.

            This is a simplified version with restricted error handling
            """

            def __init__(self):  # noqa
                self._fitted = False

            def fit(self, X: pd.Series, y: pd.Series | None = None, **fit_params) -> Self:  # noqa: ANN003
                """Persist observed categories during fitting."""
                if isinstance(X, pd.DataFrame):
                    raise ValueError("CategoryTransformer only accepts single column series")
                X = X.astype("category")
                self._categories = X.cat.categories
                self._fitted = True
                return self

            def transform(self, X: pd.Series, y: pd.DataFrame | pd.Series | None = None) -> pd.DataFrame:
                """Transform columns to categorical type using the categories observed during fitting."""
                # Seems that column transformer passes series but expected dataframes
                return X.astype("category").cat.set_categories(self._categories).to_frame()

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

    def log_model(self, dataset_type: Literal["PandasDataset", "SparkDataset"] = "SparkDataset") -> None:
        """Log the model using MLflow."""
        mlflow.set_experiment(self.experiment_name_fe)
        with mlflow.start_run(tags=self.tags) as run:
            model = self.pipeline.named_steps["lgbm"]
            mlflow.log_dict({"best_valid_loss": model.best_score_}, "best_score.json")
            mlflow.log_metric("best_iteration", model.best_iteration_)
            self.run_id = run.info.run_id

            y_pred_proba = self.pipeline.predict_proba(self.X_valid)
            y_pred = self.pipeline.predict(self.X_valid)

            # Evaluate metrics
            ll = log_loss(self.y_valid, y_pred_proba)
            report = classification_report(self.y_valid, y_pred)

            logger.info(f"📊 Test log loss: {ll}")

            # Log parameters and metrics
            mlflow.log_param("model_type", "LightGBM with preprocessing")
            mlflow.log_metric("test_log_loss", ll)  # noqa
            mlflow.log_dict({"report": report}, "report.json")

            # Log the model
            signature = infer_signature(model_input=self.X_valid, model_output=y_pred_proba)

            self.fe.log_model(
                model=self.pipeline,
                flavor=mlflow.sklearn,
                artifact_path="lightgbm-pipeline-model-fe",
                signature=signature,
                pyfunc_predict_fn="predict_proba",
                training_set=self.train_set,
            )

    def register_model(self) -> None:
        """Register model in Unity Catalog."""
        logger.info("🔄 Registering the model in UC...")
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/lightgbm-pipeline-model-fe",
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

    def load_latest_model_and_predict(self, X: DataFrame) -> DataFrame:
        """Load the latest model from MLflow (alias=latest-model) and make predictions.

        Alias latest is not allowed -> we use latest-model instead as an alternative.

        :param input_data: Pandas DataFrame containing input features for prediction.
        :return: Pandas DataFrame with predictions.
        """
        model_uri = f"models:/{self.model_name}@latest-model"

        # Make predictions
        predictions = self.fe.score_batch(model_uri=model_uri, df=X)

        # Return predictions as a DataFrame
        return predictions
