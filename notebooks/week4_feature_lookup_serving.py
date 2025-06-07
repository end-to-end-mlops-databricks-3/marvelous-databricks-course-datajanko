# Databricks notebook source
# Databricks notebook source
%pip install -e ..
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

# Documented FeatureEngineering Flow

# import mlflow
# from databricks import feature_engineering
# fe = feature_engineering.FeatureEngineeringClient()
# mlflow.set_registry_uri("databricks-uc")

# from databricks.ml_features.online_store_spec import AmazonDynamoDBSpec
# aws_dynamo_db_spec = AmazonDynamoDBSpec(region='eu-west-1', access_key_id=dbutils.secrets.get(scope="mlops_course", key="aws_access_key_id"), secret_access_key=dbutils.secrets.get(scope="mlops_course", key="aws_secret_access_key"))

# fe.publish_table(name=overall_feature_table_name, online_store=aws_dynamo_db_spec)
# fe.publish_table(name=per_map_feature_table_name, online_store=aws_dynamo_db_spec)

# COMMAND ----------



# COMMAND ----------

overall_feature_table_name = f"{config.catalog_name}.{config.schema_name}.overall_winning_shares_fe_v2"
per_map_feature_table_name = f"{config.catalog_name}.{config.schema_name}.per_map_winning_shares_fe_v2"

# COMMAND ----------

from databricks.sdk import WorkspaceClient
workspace = WorkspaceClient()

# COMMAND ----------

overall_spec = OnlineTableSpec(
            primary_key_columns=["team_name", "date"],
            source_table_full_name=overall_feature_table_name,
            run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
            perform_full_copy=False,
            timeseries_key="date"
        )
per_map_spec = OnlineTableSpec(
            primary_key_columns=["map_name", "team_name", "date"],
            source_table_full_name=per_map_feature_table_name,
            run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
            perform_full_copy=False,
            timeseries_key="date"
        )
workspace.online_tables.create(name=f"{overall_feature_table_name}_online", spec=overall_spec)
workspace.online_tables.create(name=f"{per_map_feature_table_name}_online", spec=per_map_spec)

# COMMAND ----------

# Need to check models.predict first

# from databricks.sdk import WorkspaceClient
# from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

# # Create endpoint
# endpoint_name = "cs-go-feature-serving"
# endpoint_exists = any(item.name == endpoint_name for item in workspace.serving_endpoints.list())
# model_name=f"{catalog_name}.{schema_name}.cs_go_model_feature_lookup"
# model_version=9
# workspace = WorkspaceClient()
# served_entities = [
#     ServedEntityInput(
#         entity_name=model_name,
#         scale_to_zero_enabled=True,
#         workload_size="Small",
#         entity_version=model_version,
#         environment_vars={
#                     "aws_access_key_id": "{{secrets/mlops_course/aws_access_key_id}}",
#                     "aws_secret_access_key": "{{secrets/mlops_course/aws_secret_access_key}}",
#                     "region_name": "eu-west-1",
#                     }
#     )
# ]
# if not endpoint_exists:
#     workspace.serving_endpoints.create(
#         name=endpoint_name,
#         config=EndpointCoreConfigInput(
#             served_entities=served_entities,
#         ),
#     )
# else:
#     workspace.serving_endpoints.update_config(
#         name=endpoint_name,
#         served_entities=served_entities,
#     )

# COMMAND ----------

# def call_endpoint(record):
#     """
#     Calls the model serving endpoint with a given input record.
#     """
#     serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/{endpoint_name}/invocations"

#     response = requests.post(
#         serving_endpoint,
#         headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
#         json={"dataframe_records": record},
#     )
#     return response.status_code, response.text

# COMMAND ----------

# test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set")

# COMMAND ----------

# sample = test_set.drop('map_winner').toPandas().sample(n=10).to_dict(orient='records')

# COMMAND ----------

# dataframe_records = [[record] for record in sample]

# COMMAND ----------

# note that the data contains a timestamp column -> not serializable
# dataframe_records[0]

# COMMAND ----------

# import pandas as pd

# COMMAND ----------

# record = [{'date': '2012-03-14',
#   'team_1': 'Astralis',
#   'team_2': 'Paradox',
#   'map_name': 'Mirage',
#   'starting_ct': 1,
#   'rank_1': 12,
#   'rank_2': 10}]

# COMMAND ----------

# a,b = call_endpoint(record)

# COMMAND ----------


