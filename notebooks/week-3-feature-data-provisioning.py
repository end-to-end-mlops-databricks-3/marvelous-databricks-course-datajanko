# Databricks notebook source
# MAGIC %pip install -e ..
# MAGIC %pip install git+https://github.com/end-to-end-mlops-databricks-3/marvelous@0.1.0

# COMMAND ----------

# restart python
%restart_python

# COMMAND ----------

import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent / "src"))

# COMMAND ----------

import os
from datetime import datetime

import mlflow
import pyspark.sql.functions as F
from databricks import feature_engineering
from databricks.feature_engineering import FeatureLookup
from dotenv import load_dotenv
from marvelous.common import is_databricks
from pyspark.sql import SparkSession
from pyspark.sql.window import Window

from mlops_course.config import ProjectConfig

# COMMAND ----------

if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")


config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
fe = feature_engineering.FeatureEngineeringClient()

train_set = spark.table(f"{config.catalog_name}.{config.schema_name}.train_set")
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set")

# COMMAND ----------

overall_feature_table_name = f"{config.catalog_name}.{config.schema_name}.overall_winning_shares_fe_v2"
per_map_feature_table_name = f"{config.catalog_name}.{config.schema_name}.per_map_winning_shares_fe_v2"

# COMMAND ----------

map_winners_long = (
    train_set.union(test_set)
    .withColumn("id", F.monotonically_increasing_id())
    .unpivot(["id", "date", "map_name", "map_winner"], ["team_1", "team_2"], "team_id", "team_name")
    .withColumn(
        "has_won",
        (F.substring("team_id", -1, 1).cast("int") == F.col("map_winner")).cast("float"),
    )
)

# COMMAND ----------

wins_per_day = map_winners_long.groupBy("date", "team_name").agg(
    F.sum("has_won").alias("has_won"), F.count("*").alias("matches")
)
window_for_wins_per_day = Window.orderBy("date").partitionBy("team_name").rowsBetween(-10, -1)
wins_per_map_and_day = map_winners_long.groupBy("date", "map_name", "team_name").agg(
    F.sum("has_won").alias("has_won"), F.count("*").alias("matches")
)
window_for_wins_per_map_and_day = Window.orderBy("date").partitionBy("map_name", "team_name").rowsBetween(-10, -1)


# COMMAND ----------

# It's super easy to adjust this computation to compute win shares per map, against a specific team etc.
winning_shares = (
    wins_per_day.withColumn(
        "last_10_match_day_winshare",
        F.ifnull(
            F.sum("has_won").over(window_for_wins_per_day) / F.sum(F.col("matches")).over(window_for_wins_per_day),
            F.lit(-1.0),
        ),
    )
    .withColumn(
        "last_10_days_match_count",
        F.ifnull(F.sum("matches").over(window_for_wins_per_day), F.lit(0.0)),
    )
    .drop("has_won", "matches")
)
winning_shares_per_map = (
    wins_per_map_and_day.withColumn(
        "last_10_match_day_win_share_for_map",
        F.ifnull(
            F.sum("has_won").over(window_for_wins_per_map_and_day) / F.sum("matches").over(window_for_wins_per_day),
            F.lit(-1.0),
        ),
    )
    .withColumn(
        "last_10_days_match_count_for_map", F.ifnull(F.sum("matches").over(window_for_wins_per_day), F.lit(0.0))
    )
    .drop("has_won", "matches")
)

# COMMAND ----------

fe.create_table(
    name=overall_feature_table_name, primary_keys=["team_name", "date"], timeseries_column="date", df=winning_shares
)
fe.create_table(
    name=per_map_feature_table_name,
    primary_keys=["map_name", "team_name", "date"],
    timeseries_column="date",
    df=winning_shares_per_map,
)

# COMMAND ----------

display(winning_shares)  # noqa

# COMMAND ----------

spark.sql(f"ALTER TABLE {overall_feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
spark.sql(f"ALTER TABLE {per_map_feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# COMMAND ----------

query_df = spark.createDataFrame(
    [
        (datetime(2023, 1, 1), "not_a_team_1", "not_a_team_2", "Mirage", 1),
        (datetime(2016, 10, 10), "BIG", "Astralis", "Overpass", 2),
        (datetime(2017, 10, 10), "BIG", "Astralis", "Overpass", 1),
    ],
    schema="date timestamp, team_1 string, team_2 string, map_name string, map_winner int",
)

# COMMAND ----------

ts = fe.create_training_set(
    df=query_df,
    label="map_winner",
    feature_lookups=[
        FeatureLookup(
            table_name=overall_feature_table_name,
            lookup_key=["team_1"],
            timestamp_lookup_key="date",
            feature_names=["last_10_match_day_winshare", "last_10_days_match_count"],
            rename_outputs={
                "last_10_match_day_winshare": "team_1_last_10_match_day_winshare",
                "last_10_days_match_count": "team_1_last_10_match_day_match_count",
            },
        ),
        FeatureLookup(
            table_name=per_map_feature_table_name,
            lookup_key=["map_name", "team_1"],
            timestamp_lookup_key="date",
            feature_names=["last_10_match_day_win_share_for_map", "last_10_days_match_count_for_map"],
            rename_outputs={
                "last_10_match_day_win_share_for_map": "team_1_last_10_match_day_win_share_for_map",
                "last_10_days_match_count_for_map": "team_1_last_10_match_day_match_count_for_map",
            },
        ),
        FeatureLookup(
            table_name=overall_feature_table_name,
            lookup_key=["team_2"],
            timestamp_lookup_key="date",
            feature_names=["last_10_match_day_winshare", "last_10_days_match_count"],
            rename_outputs={
                "last_10_match_day_winshare": "team_2_last_10_match_day_winshare",
                "last_10_days_match_count": "team_2_last_10_match_day_match_count",
            },
        ),
        FeatureLookup(
            table_name=per_map_feature_table_name,
            lookup_key=["map_name", "team_2"],
            timestamp_lookup_key="date",
            feature_names=["last_10_match_day_win_share_for_map", "last_10_days_match_count_for_map"],
            rename_outputs={
                "last_10_match_day_win_share_for_map": "team_2_last_10_match_day_win_share_for_map",
                "last_10_days_match_count_for_map": "team_2_last_10_match_day_match_count_for_map",
            },
        ),
    ],
    exclude_columns=["date"],
)

# COMMAND ----------

ts.load_df().display()

# COMMAND ----------


