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

os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

catalog_name = config.catalog_name
schema_name = config.schema_name
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")


# COMMAND ----------

model_name = 'cs_go_model_feature_lookup'

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")
model_uri = f"models:/{catalog_name}.{schema_name}.{model_name}@latest-model"

        # Make predictions
predictions = fe.score_batch(model_uri=model_uri, df=test_set)


# COMMAND ----------

predictions_to_serve = predictions.select('date', 'team_1', 'team_2', 'map_name',  'prediction').dropDuplicates()

# COMMAND ----------

predictions_to_serve.display()

# COMMAND ----------

feature_table_name = f"{catalog_name}.{schema_name}.served_predictions"

# COMMAND ----------

fe.create_table(
    name=feature_table_name,
    df=predictions_to_serve,
    primary_keys = ['date', 'team_1', 'team_2', 'map_name'] )

# COMMAND ----------

spark.sql(f"""
          ALTER TABLE {feature_table_name}
          SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """)


# COMMAND ----------

feature_spec_name = f"{catalog_name}.{schema_name}.feature_lookup_spec"

# COMMAND ----------

from databricks.feature_engineering import FeatureLookup
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
workspace=WorkspaceClient()

spec = OnlineTableSpec(
            primary_key_columns=['date', 'team_1', 'team_2', 'map_name'],  # Feature lookup key
            source_table_full_name=feature_table_name,
            run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict(
                {"triggered": "true"}
            ),  # Sets the policy to update the online table when triggered (not on a schedule)
            perform_full_copy=False,  # Performs incremental updates instead of full snapshot
        )
workspace.online_tables.create(name=f"{feature_table_name}_online", spec=spec)


features = [
            FeatureLookup(
                table_name=feature_table_name,
                lookup_key=['date', 'team_1', 'team_2', 'map_name'],
            )
        ]
fe.create_feature_spec(name=feature_spec_name, features=features, exclude_columns=None)



# COMMAND ----------

endpoint_name = f"cs_go_feature_lookup_endpoint"
 
endpoint_exists = any(item.name == endpoint_name for item in workspace.serving_endpoints.list())

served_entities = served_entities = [
            ServedEntityInput(
                entity_name=feature_spec_name,  scale_to_zero_enabled=True,
        workload_size="Small",)
        ]

if not endpoint_exists:
    workspace.serving_endpoints.create(
        name=endpoint_name,
        config=EndpointCoreConfigInput(
                served_entities=served_entities,
                ),
            )
else:
    workspace.serving_endpoints.update_config(name=endpoint_name, served_entities=served_entities)

# COMMAND ----------

start_time = time.time()
serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/{endpoint_name}/invocations"
response = requests.post(
    f"{serving_endpoint}",
    headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
    json={"dataframe_records": [{"date": "2020-03-06", 'team_1': 'SpeedRunners', 'team_2': 'HellRaisers', 'map_name':'Dust2'}]},
)

end_time = time.time()
execution_time = end_time - start_time

print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")


# COMMAND ----------
# another way to call the endpoint

# response = requests.post(
#     f"{serving_endpoint}",
#     headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
#     json={"dataframe_split": {"columns": ["Id"], "data": [["182"]]}},
# )

# COMMAND ----------


