# Databricks notebook source
# MAGIC %md
# MAGIC # CS Go winning probability exercise
# MAGIC
# MAGIC This notebook demonstrates how to predict which of two teams is going to in CS:GO on a specific map using a CS:GO matches dataset from kaggle. We'll go through the process of loading data, preprocessing, model creation, and visualization of results.
# COMMAND ----------
import yaml
import pandas as pd
from pyspark.sql import SparkSession

with open("../project_config.yml", "r") as file:
    config = yaml.safe_load(file)

catalog_name = config["dev"]["catalog_name"]
schema_name = config["dev"]["schema_name"]

raw_data_columns = config["raw_data_columns"]

spark = SparkSession.builder.getOrCreate()
# Works both locally and in a Databricks environment
filepath = "../data/results.csv"
# Load the data
raw_df = pd.read_csv(filepath)[raw_data_columns]
# COMMAND ----------
# Typisation
typed_df = raw_df.assign(date=lambda df: pd.to_datetime(df['date'], errors='coerce'), 
          team_1=lambda df:df['team_1'].astype('category'),
          team_2=lambda df:df['team_2'].astype('category'),
          map_name = lambda df:df['_map'].astype('category')
          ).drop('_map', axis=1)
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
