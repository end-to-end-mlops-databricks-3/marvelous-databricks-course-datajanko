# Databricks notebook source
# MAGIC %pip install -e ..

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC # Test Basic Model

# COMMAND ----------

import json

# COMMAND ----------

import mlflow
model_uri = 'runs:/0c35610aa0644a56a106bfdf19b5a8dc/lightgbm-pipeline-model'
input_data = [{'date': '2012-03-14',
  'team_1': 'Astralis',
  'team_2': 'Paradox',
  'map_name': 'Mirage',
  'starting_ct': 1,
  'rank_1': 12,
  'rank_2': 10}]

mlflow.models.predict(
    model_uri=model_uri,
    input_data=input_data,
)

# COMMAND ----------

# this cell is not suficient, as there are odd erros:
# MlflowException: Input data could not be serialized to json.
# and
# --> 240     from mlflow.pyfunc.scoring_server import SUPPORTED_FORMATS, SUPPORTED_LLM_FORMATS
#     242     if isinstance(input_data, dict) and any(
#     243         key in input_data for key in SUPPORTED_FORMATS | SUPPORTED_LLM_FORMATS
#     244     ):
# File /databricks/python/lib/python3.11/site-packages/mlflow/pyfunc/scoring_server/__init__.py:24
#      22 from typing import Dict, NamedTuple, Optional, Tuple
# ---> 24 import flask
#      26 from mlflow.environment_variables import MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT
# ModuleNotFoundError: No module named 'flask'
# import mlflow
# model_uri = 'runs:/0516110e223a45ddaff1bd234284ead4/lightgbm-pipeline-model-fe'
# input_data = [{'date': '2012-03-14',
#   'team_1': 'Astralis',
#   'team_2': 'Paradox',
#   'map_name': 'Mirage',
#   'starting_ct': 1,
#   'rank_1': 12,
#   'rank_2': 10}]

# mlflow.models.predict(
#     model_uri=model_uri,
#     input_data=input_data,
# )

# COMMAND ----------


