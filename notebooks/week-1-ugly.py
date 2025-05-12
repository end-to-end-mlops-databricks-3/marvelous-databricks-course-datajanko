# Databricks notebook source
# MAGIC %md
# MAGIC # CS Go winning probability exercise
# MAGIC
# MAGIC This notebook demonstrates how to predict which of two teams is going to in CS:GO on a specific map using a CS:GO matches dataset from kaggle. We'll go through the process of loading data, preprocessing, model creation, and visualization of results.
# COMMAND ----------
import pandas as pd
import yaml
from pyspark.sql import SparkSession

with open("../project_config.yml") as file:
    config = yaml.safe_load(file)

catalog_name = config["dev"]["catalog_name"]
schema_name = config["dev"]["schema_name"]

raw_data_columns = config["raw_data"]["columns"]

spark = SparkSession.builder.getOrCreate()
# Works both locally and in a Databricks environment
filepath = "../data/results.csv"
# Load the data
raw_df = pd.read_csv(filepath)[raw_data_columns]
# COMMAND ----------
# Typisation
typed_df = raw_df.assign(
    date=lambda df: pd.to_datetime(df["date"], errors="coerce"),
    team_1=lambda df: df["team_1"].astype("category"),
    team_2=lambda df: df["team_2"].astype("category"),
    map_name=lambda df: df["_map"].astype("category"),
).drop("_map", axis=1)
# COMMAND ----------
# We're going to use this typed df later to generate historical features, like
# winshare in the last 10 games
# Winshare per map
# winshare per map on the last 10 games on that map
# Average score per map as t and CT
# average score per map as t and CT in last 10 games
# winshare against oponent
# average score against oponent
#
# This is going to be a lot of overhead, so we do not know whether we really achieve this
# We're planning to use lightgbm which natively handles NaN's and we do not need special treatment for NaNs
#
# Data set needs to be split by time. We could even simulate the arrival of new data and update the features mentioned above
# To generate those features, auxiliary tables need to be created from essenitally the raw data
# We will do a train an test split later. This will look like data leakage could occur, but we can prevent this if we're careful
# Moreover, this is just an example show case.


# COMMAND ----------
date_col = "date"
target = "map_winner"
training_data_columns = ["team_1", "team_2", "map_name", "starting_ct", "rank_1", "rank_2"]

data_df = typed_df[[date_col] + training_data_columns + [target]]

# COMMAND ----------
# Simplest possible temporal split. Not doing a temporal split could lead to data leakage as discussed in discord
last_training_day = pd.Timestamp("2019-12-31")

train_set = data_df.query("date <= @last_training_day")
test_set = data_df.query("date > @last_training_day")


# COMMAND ----------
# Persist tables on databricks
# I think I will not write this more than once, but I might need to change the schema. Doing it like this will ensure, that the notebook will always work
spark.createDataFrame(typed_df).write.mode("overwrite").option("overwriteSchema", True).saveAsTable(
    f"{catalog_name}.{schema_name}.parsed_data"
)

# I'm not sure yet about how I want to include the timestamps here. Also there might be more than one split to provide cross-validation
spark.createDataFrame(train_set).write.mode("overwrite").saveAsTable(f"{catalog_name}.{schema_name}.train_set")

spark.createDataFrame(test_set).write.mode("overwrite").saveAsTable(f"{catalog_name}.{schema_name}.test_set")
# COMMAND ----------
