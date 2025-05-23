# Databricks notebook source

# % pip install -e ..
# %restart_python

# from pathlib import Path
# import sys
# sys.path.append(str(Path.cwd().parent / 'src'))
# COMMAND ----------

import os

import mlflow
from dotenv import load_dotenv
from marvelous.common import is_databricks
from pyspark.sql import SparkSession

from mlops_course.config import ProjectConfig, Tags
from mlops_course.models.basic_model import BasicModel

if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

# COMMAND ----------
import mlflow
from lightgbm import register_logger
from loguru import logger

# COMMAND ----------
register_logger(logger)

config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
spark = SparkSession.builder.getOrCreate()

tags = Tags(**{"git_sha": "abcd12345", "branch": "week2"})

basic_model = BasicModel(config, tags, spark)

# COMMAND ----------
basic_model.load_data()
# COMMAND ----------
basic_model.create_pipeline()
# COMMAND ----------
basic_model.train()
# COMMAND ----------
basic_model.log_model()

# Retrieve dataset for the current run
basic_model.retrieve_current_run_dataset()

# COMMAND ----------
# Retrieve metadata for the current run
basic_model.retrieve_current_run_metadata()

# COMMAND ----------
# Register model
basic_model.register_model()

# COMMAND ----------
# Predict on the test set

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

X_test = test_set.drop(config.selection.target).toPandas()

predictions_df = basic_model.load_latest_model_and_predict(X_test)
predictions_df  # noqa: B018
