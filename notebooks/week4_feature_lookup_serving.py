# Databricks notebook source
# Databricks notebook source
%pip install /Volumes/mlops_dev/jankoch8/data/mlops_course-0.1.0-py3-none-any.whl
%pip install git+https://github.com/end-to-end-mlops-databricks-3/marvelous@0.1.0

# COMMAND ----------

# COMMAND ----------
%restart_python

# COMMAND ----------

import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent / "src"))

# COMMAND ----------

# COMMAND ----------
import os
import time
from typing import Dict, List

import requests
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession
from mlops_course.config import ProjectConfig

# spark session

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# COMMAND ----------

config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

# COMMAND ----------



# COMMAND ----------

from databricks.sdk.service.catalog import (
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

# COMMAND ----------

os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

import mlflow
from databricks import feature_engineering
fe = feature_engineering.FeatureEngineeringClient()
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

overall_feature_table_name = f"{config.catalog_name}.{config.schema_name}.overall_winning_shares"
per_map_feature_table_name = f"{config.catalog_name}.{config.schema_name}.per_map_winning_shares"

# COMMAND ----------

from databricks.sdk import WorkspaceClient
workspace = WorkspaceClient()

# COMMAND ----------

# overall_spec = OnlineTableSpec(
#             primary_key_columns=["date", "team_name"],
#             source_table_full_name=overall_feature_table_name,
#             run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
#             perform_full_copy=False,
#             timeseries_key="date"
#         )
# per_map_spec = OnlineTableSpec(
#             primary_key_columns=["date", "map_name", "team_name"],
#             source_table_full_name=per_map_feature_table_name,
#             run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
#             perform_full_copy=False,
#             timeseries_key="date"
#         )
# workspace.online_tables.create(name=f"{overall_feature_table_name}_online", spec=overall_spec)
# workspace.online_tables.create(name=f"{per_map_feature_table_name}_online", spec=per_map_spec)

# COMMAND ----------

endpoint_name = "cs-go-feature-serving"

# COMMAND ----------

# from databricks.ml_features.online_store_spec import AmazonDynamoDBSpec
# aws_dynamo_db_spec = AmazonDynamoDBSpec(region='eu-west-1', access_key_id=dbutils.secrets.get(scope="mlops_course", key="aws_access_key_id"), secret_access_key=dbutils.secrets.get(scope="mlops_course", key="aws_secret_access_key"))

# fe.drop_online_table(name=overall_feature_table_name, online_store=aws_dynamo_db_spec)
# fe.drop_online_table(name=per_map_feature_table_name, online_store=aws_dynamo_db_spec)

# COMMAND ----------



# COMMAND ----------

endpoint_exists = any(item.name == endpoint_name for item in workspace.serving_endpoints.list())


# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

# Create endpoint
endpoint_name = "cs-go-feature-serving"
endpoint_exists = any(item.name == endpoint_name for item in workspace.serving_endpoints.list())
model_name=f"{catalog_name}.{schema_name}.cs_go_model_feature_lookup"
model_version=5
workspace = WorkspaceClient()
served_entities = [
    ServedEntityInput(
        entity_name=model_name,
        scale_to_zero_enabled=True,
        workload_size="Small",
        entity_version=model_version,
        environment_vars={
                    "aws_access_key_id": "{{secrets/mlops_course/aws_access_key_id}}",
                    "aws_secret_access_key": "{{secrets/mlops_course/aws_secret_access_key}}",
                    "region_name": "eu-west-1",
                    }
    )
]
if not endpoint_exists:
    workspace.serving_endpoints.create(
        name=endpoint_name,
        config=EndpointCoreConfigInput(
            served_entities=served_entities,
        ),
    )
else:
    workspace.serving_endpoints.update_config(
        name=endpoint_name,
        served_entities=served_entities,
    )

# COMMAND ----------

os.environ["aws_access_key_id"]

# COMMAND ----------


