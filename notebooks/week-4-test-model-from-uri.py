# Databricks notebook source
# MAGIC %pip install -e ..

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC # Test Basic Model

# COMMAND ----------

# COMMAND ----------
import mlflow
import pandas as pd

model_uri = "runs:/0c35610aa0644a56a106bfdf19b5a8dc/lightgbm-pipeline-model"
input_data = [
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
# Learning: Using the raw list, leads to a strange conversion/schema violation error
new_data = pd.DataFrame(input_data)

mlflow.models.predict(
    model_uri=model_uri,
    input_data=new_data,
)

# COMMAND ----------
