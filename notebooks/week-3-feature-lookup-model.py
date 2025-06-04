# Databricks notebook source
# MAGIC %pip install -e ..
# MAGIC %pip install git+https://github.com/end-to-end-mlops-databricks-3/marvelous@0.1.0

# COMMAND ----------

# restart python
# %restart_python #noqa

# COMMAND ----------

import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent / "src"))


# COMMAND ----------

from lightgbm import register_logger
from loguru import logger

from mlops_course.models.feature_lookup_model import FeatureLookupModel, ProjectConfig, SparkSession, Tags

register_logger(logger)

config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
spark = SparkSession.builder.getOrCreate()

tags = Tags(**{"git_sha": "abcd12345", "branch": "week3"})

# basic_model = BasicModel(config, tags, spark)

# COMMAND ----------

flm = FeatureLookupModel(config, tags, spark)

# COMMAND ----------

flm.load_data()
flm.create_pipeline()
flm.train()
flm.log_model()
flm.register_model()

# COMMAND ----------

flm.load_latest_model_and_predict(flm.test_set).display()
