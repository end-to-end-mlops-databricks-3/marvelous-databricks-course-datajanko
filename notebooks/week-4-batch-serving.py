# Databricks notebook source
# %pip install -e ..
# %pip install git+https://github.com/end-to-end-mlops-databricks-3/marvelous@0.1.0


# COMMAND ----------
# %restart_python

# COMMAND ----------

import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent / "src"))


# COMMAND ----------
import os
import time

import requests
from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from mlops_course.config import ProjectConfig
from mlops_course.serving.feature_serving import FeatureServing

# spark session

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

catalog_name = config.catalog_name
schema_name = config.schema_name
model_name = "cs_go_model_feature_lookup"


# COMMAND ----------
# populate a feature table
feature_table_name = f"{catalog_name}.{schema_name}.served_predictions"
fe = FeatureEngineeringClient()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")
model_uri = f"models:/{catalog_name}.{schema_name}.{model_name}@latest-model"

predictions = fe.score_batch(model_uri=model_uri, df=test_set)
predictions_to_serve = predictions.select("date", "team_1", "team_2", "map_name", "prediction").dropDuplicates()
fe.create_table(name=feature_table_name, df=predictions_to_serve, primary_keys=["date", "team_1", "team_2", "map_name"])

spark.sql(f"""
          ALTER TABLE {feature_table_name}
          SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """)


# COMMAND ----------

feature_spec_name = f"{catalog_name}.{schema_name}.feature_lookup_spec"

endpoint_name = "cs_go_feature_lookup_endpoint"
# COMMAND ----------
fs = FeatureServing(
    feature_table_name=feature_table_name, feature_spec_name=feature_spec_name, endpoint_name=endpoint_name
)
fs.create_online_table().create_feature_spec().deploy_or_update_serving_endpoint()

# COMMAND ----------

start_time = time.time()
serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/{endpoint_name}/invocations"
response = requests.post(
    f"{serving_endpoint}",
    headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
    json={
        "dataframe_records": [
            {"date": "2020-03-06", "team_1": "SpeedRunners", "team_2": "HellRaisers", "map_name": "Dust2"}
        ]
    },
)

end_time = time.time()
execution_time = end_time - start_time

print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")
