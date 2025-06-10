# Databricks notebook source
# Databricks notebook source
# %pip install -e ..
# %pip install git+https://github.com/end-to-end-mlops-databricks-3/marvelous@0.1.0

# COMMAND ----------

# COMMAND ----------
# %restart_python

# COMMAND ----------

import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path.cwd().parent / "src"))

# COMMAND ----------

# COMMAND ----------
import os

import requests
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from mlops_course.config import ProjectConfig
from mlops_course.serving.model_serving import ModelServing

# spark session

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# COMMAND ----------

config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")


# COMMAND ----------

os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")  # noqa

# COMMAND ----------

catalog_name = config.catalog_name
schema_name = config.schema_name
model_name = f"{catalog_name}.{schema_name}.cs_go_model_basic"
endpoint_name = "cs-go-basoc-feature-serving"
ms = ModelServing(model_name=model_name, endpoint_name="cs-go-basoc-feature-serving")

# COMMAND ----------


def call_endpoint(record: dict[str, Any]) -> tuple[Any, Any]:
    """Call the model serving endpoint with a given input record."""
    serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/{endpoint_name}/invocations"

    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record},
    )
    return response.status_code, response.text


# COMMAND ----------

record = [
    {
        "date": "2012-03-14",
        "team_1": "Astralis",
        "team_2": "Paradox",
        "map_name": "Mirage",
        "starting_ct": 1,
        "rank_1": 12,
        "rank_2": 10,
    }
]

# COMMAND ----------

call_endpoint(record)
