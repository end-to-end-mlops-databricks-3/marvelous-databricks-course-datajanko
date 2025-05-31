# Databricks notebook source
# MAGIC %pip install -e ..
# MAGIC %pip install git+https://github.com/end-to-end-mlops-databricks-3/marvelous@0.1.0

# COMMAND ----------

#restart python
%restart_python

# COMMAND ----------

from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent / 'src'))

# COMMAND ----------

from typing import Protocol
from mlops_course.config import ProjectConfig, Tags
from pyspark.sql import SparkSession

class Trainable(Protocol):
    config: ProjectConfig
    tags: Tags
    spark: SparkSession

    def load_data(self) -> None:
        ...
    def create_pipeline(self) -> None:
        ...
    def train(self) -> None:
        ...
    def log_model(self) -> None:
        ...
    def register_model(self) -> None:
        ...

class ModelTrainer:
    def __init__(self, trainable:Trainable):
        self.trainable = trainable

    def run(self) -> None:
        self.trainable.load_data()
        self.trainable.create_pipeline()
        self.trainable.train()
        self.trainable.log_model()
        self.trainable.register_model()

# COMMAND ----------

"""Feature Lookup model implementation.

infer_signature (from mlflow.models) → Captures input-output schema for model tracking.

num_features → List of numerical feature names.
cat_features → List of categorical feature names.
target → The column to predict.
parameters → Hyperparameters for LightGBM.
catalog_name, schema_name → Database schema names for Databricks tables.
"""

from typing import Any, Literal

import mlflow
import mlflow.data
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from loguru import logger
from mlflow import MlflowClient
from mlflow.data.dataset_source import DatasetSource
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, log_loss
from sklearn.pipeline import Pipeline

from mlops_course.config import ProjectConfig, Tags
from mlops_course.transformers import CategoryTransformer


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
        self.train_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        self.train_set_pandas = self.train_set.toPandas()
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set")
        self.test_set_pandas = self.test_set.toPandas()
        self.data_version = "0"  # describe history -> retrieve

        train = self.train_set_pandas.query("date < @self.validation_start")
        valid = self.train_set_pandas.query("date >= @self.validation_start")
        self.X_train = train[self.features]
        self.y_train = train[self.target]
        self.X_valid = valid[self.features]
        self.y_valid = valid[self.target]
        self.X_test = self.test_set_pandas[self.features]
        self.y_test = self.test_set_pandas[self.target]

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

    def log_model(self, dataset_type: Literal["PandasDataset", "SparkDataset"] = "SparkDataset") -> None:
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
            signature = infer_signature(model_input=self.X_train, model_output=y_pred_proba)
            if dataset_type == "PandasDataset":
                dataset = mlflow.data.from_pandas(
                    self.train_set_pandas,
                    name="train_set",
                )
            elif dataset_type == "SparkDataset":
                dataset = mlflow.data.from_spark(
                    self.train_set,
                    table_name=f"{self.catalog_name}.{self.schema_name}.train_set",
                    version=self.data_version,
                )
            else:
                raise ValueError("Unsupported dataset type.")
            mlflow.log_input(dataset, context="training")
            mlflow.sklearn.log_model(
                sk_model=self.pipeline,
                artifact_path="lightgbm-pipeline-model",
                signature=signature,
                pyfunc_predict_fn="predict_proba",
            )

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


# COMMAND ----------

import mlflow
from lightgbm import register_logger
from loguru import logger
from mlops_course.models.basic_model import BasicModel

register_logger(logger)

config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
spark = SparkSession.builder.getOrCreate()

tags = Tags(**{"git_sha": "abcd12345", "branch": "week2"})

basic_model = BasicModel(config, tags, spark)

# COMMAND ----------

flm = FeatureLookupModel(config, tags, spark)

# COMMAND ----------

basic_trainer = ModelTrainer(basic_model)
basic_trainer.run()

# COMMAND ----------

train_set = spark.table(f"{flm.catalog_name}.{flm.schema_name}.train_set")
# train_set_pandas = train_set.toPandas()
test_set = spark.table(f"{flm.catalog_name}.{flm.schema_name}.test_set")
# test_set_pandas = test_set.toPandas()
data_version = "0"  # describe history -> retrieve

# train = train_set_pandas.query("date < @self.validation_start")
# valid = train_set_pandas.query("date >= @self.validation_start")
# X_train = train[flm.features]
# y_train = train[self.target]
# X_valid = valid[self.features]
# y_valid = valid[self.target]
# X_test = self.test_set_pandas[self.features]
# y_test = self.test_set_pandas[self.target]

# COMMAND ----------

import pyspark.sql.functions as F
training_set = train_set.filter(F.col('date') < F.lit(config.validation_start_day).cast('timestamp'))
validation_set = train_set.filter(F.col('date') >= F.lit(config.validation_start_day).cast('timestamp'))

# COMMAND ----------

from databricks import feature_engineering
from databricks.feature_engineering import FeatureLookup
fe = feature_engineering.FeatureEngineeringClient()
overall_feature_table_name = f"{config.catalog_name}.{config.schema_name}.overall_winning_shares"
per_map_feature_table_name = f"{config.catalog_name}.{config.schema_name}.per_map_winning_shares"

train_set = fe.create_training_set(df=training_set, label='map_winner', feature_lookups=[
    FeatureLookup(
    table_name=overall_feature_table_name,
    lookup_key=['team_1'], timestamp_lookup_key='date', feature_names=['last_10_match_day_winshare', 'last_10_days_match_count'],
    rename_outputs={'last_10_match_day_winshare': 'team_1_last_10_match_day_winshare', 'last_10_days_match_count': 'team_1_last_10_match_day_match_count'}
), FeatureLookup(
    table_name=per_map_feature_table_name,
    lookup_key=['map_name', 'team_1'], timestamp_lookup_key='date', feature_names=['last_10_match_day_win_share_for_map', 'last_10_days_match_count_for_map'],
    rename_outputs={'last_10_match_day_win_share_for_map': 'team_1_last_10_match_day_win_share_for_map', 'last_10_days_match_count_for_map': 'team_1_last_10_match_day_match_count_for_map'}
),  FeatureLookup(
    table_name=overall_feature_table_name,
    lookup_key=['team_2'], timestamp_lookup_key='date', feature_names=['last_10_match_day_winshare', 'last_10_days_match_count'],
    rename_outputs={'last_10_match_day_winshare': 'team_2_last_10_match_day_winshare', 'last_10_days_match_count': 'team_2_last_10_match_day_match_count'}
), FeatureLookup(
    table_name=per_map_feature_table_name,
    lookup_key=['map_name', 'team_2'], timestamp_lookup_key='date', feature_names=['last_10_match_day_win_share_for_map', 'last_10_days_match_count_for_map'],
    rename_outputs={'last_10_match_day_win_share_for_map': 'team_2_last_10_match_day_win_share_for_map', 'last_10_days_match_count_for_map': 'team_2_last_10_match_day_match_count_for_map'}
)
], exclude_columns='date')

valid_set = fe.create_training_set(df=validation_set, label='map_winner', feature_lookups=[
    FeatureLookup(
    table_name=overall_feature_table_name,
    lookup_key=['team_1'], timestamp_lookup_key='date', feature_names=['last_10_match_day_winshare', 'last_10_days_match_count'],
    rename_outputs={'last_10_match_day_winshare': 'team_1_last_10_match_day_winshare', 'last_10_days_match_count': 'team_1_last_10_match_day_match_count'}
), FeatureLookup(
    table_name=per_map_feature_table_name,
    lookup_key=['map_name', 'team_1'], timestamp_lookup_key='date', feature_names=['last_10_match_day_win_share_for_map', 'last_10_days_match_count_for_map'],
    rename_outputs={'last_10_match_day_win_share_for_map': 'team_1_last_10_match_day_win_share_for_map', 'last_10_days_match_count_for_map': 'team_1_last_10_match_day_match_count_for_map'}
),  FeatureLookup(
    table_name=overall_feature_table_name,
    lookup_key=['team_2'], timestamp_lookup_key='date', feature_names=['last_10_match_day_winshare', 'last_10_days_match_count'],
    rename_outputs={'last_10_match_day_winshare': 'team_2_last_10_match_day_winshare', 'last_10_days_match_count': 'team_2_last_10_match_day_match_count'}
), FeatureLookup(
    table_name=per_map_feature_table_name,
    lookup_key=['map_name', 'team_2'], timestamp_lookup_key='date', feature_names=['last_10_match_day_win_share_for_map', 'last_10_days_match_count_for_map'],
    rename_outputs={'last_10_match_day_win_share_for_map': 'team_2_last_10_match_day_win_share_for_map', 'last_10_days_match_count_for_map': 'team_2_last_10_match_day_match_count_for_map'}
)
], exclude_columns='date')

# COMMAND ----------

train_df = train_set.load_df().fillna(-1.0, subset=['team_1_last_10_match_day_winshare', 'team_2_last_10_match_day_winshare', 'team_1_last_10_match_day_win_share_for_map', 'team_2_last_10_match_day_win_share_for_map']).fillna(0.0, subset=['team_1_last_10_match_day_match_count', 'team_2_last_10_match_day_match_count', 'team_1_last_10_match_day_match_count_for_map', 'team_2_last_10_match_day_match_count_for_map'])
valid_df = valid_set.load_df().fillna(-1.0, subset=['team_1_last_10_match_day_winshare', 'team_2_last_10_match_day_winshare', 'team_1_last_10_match_day_win_share_for_map', 'team_2_last_10_match_day_win_share_for_map']).fillna(0.0, subset=['team_1_last_10_match_day_match_count', 'team_2_last_10_match_day_match_count', 'team_1_last_10_match_day_match_count_for_map', 'team_2_last_10_match_day_match_count_for_map'])

# COMMAND ----------

config

# COMMAND ----------

category_transformers = [
            (col, CategoryTransformer().set_output(transform="pandas"), col) for col in config.selection.categories
        ]
pipe = Pipeline(
            steps=[
                (
                    "preprocessor",
                    ColumnTransformer(
                        transformers=category_transformers,
                        remainder="passthrough",
                        verbose_feature_names_out=False,
                    ).set_output(transform="pandas"),
                ),
                ("lgbm", LGBMClassifier(**config.parameters)),
            ]
        )

# COMMAND ----------

train = train_df.toPandas()
valid = valid_df.toPandas()
X_train = train.drop(config.selection.target, axis=1)
y_train = train[config.selection.target]
X_valid = valid.drop(config.selection.target,axis=1)
y_valid = valid[config.selection.target]

# COMMAND ----------

preproc = pipe.named_steps["preprocessor"]
model = pipe.named_steps["lgbm"]
X_train = preproc.fit_transform(X_train)
X_valid = preproc.transform(X_valid)
pipe.named_steps["pre_processor"] = preproc
callbacks: list[Any] = [log_evaluation(10)]
if "early_stopping" in config.fit_parameters:
    callbacks.append(early_stopping(int(config.fit_parameters["early_stopping"])))
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=callbacks)
pipe.named_steps["lgbm"] = model

# COMMAND ----------

y_pred_proba = pipe.predict_proba(X_valid)
y_pred = pipe.predict(X_valid)

# COMMAND ----------



# COMMAND ----------

mlflow.set_experiment(config.experiment_name_basic)
with mlflow.start_run(tags=tags) as run:
    run_id = run.info.run_id
    signature = infer_signature(X_valid, y_pred_proba)

    fe.log_model(
                model=pipe,
                flavor=mlflow.sklearn,
                artifact_path="lgbm-feature-lookup-model",
                training_set=train_set,
                signature=signature,
                pyfunc_predict_fn='predict_proba'
            )

# COMMAND ----------

model_name = f"{config.catalog_name}.{config.schema_name}.cs_go_model_feature_lookup"
model_name

# COMMAND ----------

run_id

# COMMAND ----------

import mlflow

model_uri = 'runs:/8296802b27864434879acd24af25e828/lgbm-feature-lookup-model'

# COMMAND ----------

tags

# COMMAND ----------

registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
            tags=tags.dict(),
        )
logger.info(f"✅ Model registered as version {registered_model.version}.")

# COMMAND ----------



predictions = fe.score_batch(model_uri=model_uri, df=training_set)

# COMMAND ----------

# First error
predictions.display()

# COMMAND ----------

validation_set.toJSON().collect()

# COMMAND ----------

#second error
import mlflow
from mlflow.models import Model

model_uri = 'runs:/1ba34538ee294cbab9ae6f3288e794b4/lightgbm-pipeline-model-fe'
# The model is logged with an input example
pyfunc_model = mlflow.pyfunc.load_model(model_uri)


# Verify the model with the provided input data using the logged dependencies.
# For more details, refer to:
# https://mlflow.org/docs/latest/models.html#validate-models-before-deployment
mlflow.models.predict(
    model_uri=model_uri,
    input_data=validation_set,
)

# COMMAND ----------



# COMMAND ----------


