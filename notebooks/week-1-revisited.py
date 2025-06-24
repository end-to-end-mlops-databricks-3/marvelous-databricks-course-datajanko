# Databricks notebook source

# % pip install -e ..
# %restart_python

# from pathlib import Path
# import sys
# sys.path.append(str(Path.cwd().parent / 'src'))

# COMMAND ----------
import yaml
from loguru import logger
from marvelous.logging import setup_logging
from marvelous.timer import Timer
from pyspark.sql import SparkSession

from mlops_course.config import ProjectConfig
from mlops_course.data_io import csv_data_loader, store_processed, store_train_test
from mlops_course.data_processor import basic_temporal_train_test_split, pre_processor

config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

setup_logging(log_file="logs/marvelous-1.log")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()
filepath = "../data/results.csv"

# Load and process data
with Timer() as preprocess_timer:
    processed_data = csv_data_loader(filepath, config.raw_data_columns, config.parsing).pipe(
        pre_processor, config.selection
    )

logger.info(f"Data preprocessing: {preprocess_timer}")


# COMMAND ----------
train_set, test_set = basic_temporal_train_test_split(processed_data, config.last_training_day)

# COMMAND ----------
# Persist tables on databricks
store_processed(processed_data, config, spark)
store_train_test(train_set, test_set, config, spark)

# COMMAND ----------
