from datetime import timedelta

import pyspark.sql.functions as F
import yaml
from loguru import logger
from marvelous.common import create_parser
from marvelous.timer import Timer
from pyspark.sql import SparkSession
from pyspark.sql.types import LongType

from mlops_course.config import ProjectConfig
from mlops_course.data_processor import extract_distributions, sample_matches, sample_outcomes

args = create_parser()

root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"
config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
is_test = args.is_test
is_bootstrap = args.is_bootstrap
max_date = config.max_raw_data_date
logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# Load the house prices dataset
spark = SparkSession.builder.getOrCreate()
val_offset = config.validation_size_in_days
test_offset = config.test_set_size_in_days

if is_bootstrap == 1:
    logger.info("Start Populating data sources")

    with Timer() as bootstrap_timer:
        df = spark.read.csv(
            f"/Volumes/{config.catalog_name}/{config.schema_name}/data/results.csv", header=True, inferSchema=True
        )
        processed_data = df.select(config.raw_data_columns).withColumnsRenamed(config.parsing.rename)

        spark.createDataFrame(processed_data).write.mode("overwrite").option("overwriteSchema", True).saveAsTable(
            f"{config.catalog_name}.{config.schema_name}.parsed_data"
        )
        # In case we are going with timestamp, need to add .date() or solve UTC issue differently
        test_end = max_date

        validation_end = test_end - timedelta(days=test_offset)  # noqa # type: ignore
        training_end = validation_end - timedelta(days=val_offset)  # noqa # type: ignore

        train_set = processed_data.filter(F.col("date") <= F.lit(training_end))
        validation_set = processed_data.filter(F.col("date") > F.lit(training_end)).filter(
            F.col("date") <= F.lit(validation_end)
        )
        test_set = processed_data.filter(F.col("date") > F.lit(validation_end)).filter(F.col("date") <= F.lit(test_end))

        train_set.write.mode("overwrite").saveAsTable(f"{config.catalog_name}.{config.schema_name}.train_set")
        validation_set.write.mode("overwrite").saveAsTable(f"{config.catalog_name}.{config.schema_name}.validation_set")
        test_set.write.mode("overwrite").saveAsTable(f"{config.catalog_name}.{config.schema_name}.test_set")

    logger.info(f"Bootstrapping Completed! Took: {bootstrap_timer}")

all_data = spark.read.table(f"{config.catalog_name}.{config.schema_name}.parsed_data")
original_data = all_data.filter(F.col("date") <= F.lit(max_date))

max_date = max(all_data.agg({"date": "max"}).collect()[0][0], max_date)

dists = extract_distributions(original_data.drop("date").toPandas())
# COMMAND ----------
sampled_matches = spark.createDataFrame(sample_matches(10, dists)).withColumn(
    "map_winner", F.lit(None).cast(LongType())
)
# COMMAND ----------
sampled_matches_with_date = sampled_matches.withColumn("date", F.date_add(F.lit(max_date), 1).cast("timestamp"))


if is_test == 0:
    # Generate synthetic data.
    # This is mimicking a new data arrival. In real world, this would be a new batch of data.
    # df is passed to infer schema

    sampled_matches_with_date.write.mode("append").format("delta").saveAsTable(
        f"{config.catalog_name}.{config.schema_name}.parsed_data"
    )

    logger.info("Synthetic matches generated and attached")
else:
    # Generate synthetic data
    # This is mimicking a new data arrival. This is a valid example for integration testing.
    sampled_matches_with_date.write.mode("append").format("delta").saveAsTable(
        f"{config.catalog_name}.{config.schema_name}.parsed_data"
    )
    logger.info("Test matches generated and attached.")

max_date_with_results = all_data.filter(F.col("map_winner").isNotNull()).agg({"date": "max"}).collect()[0][0]


old_train_set = spark.read.table(f"{config.catalog_name}.{config.schema_name}.train_set")
old_valid_set = spark.read.table(f"{config.catalog_name}.{config.schema_name}.validation_set")
old_test_set = spark.read.table(f"{config.catalog_name}.{config.schema_name}.test_set")

# Could pickle class to keep the state
old_train_set_max_date = old_train_set.agg({"date": "max"}).collect()[0][0]
old_valid_set_max_date = old_valid_set.agg({"date": "max"}).collect()[0][0]
old_test_set_max_date = old_test_set.agg({"date": "max"}).collect()[0][0]

test_end = max_date_with_results
validation_end = test_end - timedelta(days=test_offset)  # type:ignore
training_end = validation_end - timedelta(days=val_offset)  # type: ignore


# COMMAND ----------
train_set = all_data.filter(F.col("date") <= F.lit(training_end))
validation_set = all_data.filter(F.col("date") > F.lit(training_end)).filter(F.col("date") <= F.lit(validation_end))
test_set = all_data.filter(F.col("date") > F.lit(validation_end)).filter(F.col("date") <= F.lit(test_end))


spark.sql(
    """
          MERGE INTO {old} as p
          USING {source} as s
          ON p.date = s.date
          AND p.team_1 = s.team_1
          AND p.team_2 = s.team_2
          AND p.map_name = s.map_name
          AND p.rank_1 = s.rank_1
          AND p.rank_2 = s.rank_2
          AND p.starting_ct = s.starting_ct
          AND p.map_winner = s.map_winner
          WHEN NOT MATCHED THEN
          INSERT *
          """,
    old=old_train_set,
    source=train_set,
)
logger.info("updated training set")

spark.sql(
    """
          MERGE INTO {old} as p
          USING {source} as s
          ON p.date = s.date
          AND p.team_1 = s.team_1
          AND p.team_2 = s.team_2
          AND p.map_name = s.map_name
          AND p.rank_1 = s.rank_1
          AND p.rank_2 = s.rank_2
          AND p.starting_ct = s.starting_ct
          AND p.map_winner = s.map_winner
          WHEN NOT MATCHED BY TARGET THEN
          INSERT *
          WHEN NOT MATCHED BY SOURCE THEN
          DELETE
          """,
    old=old_valid_set,
    source=validation_set,
)
logger.info("updated validation set")

spark.sql(
    """
          MERGE INTO {old} as p
          USING {source} as s
          ON p.date = s.date
          AND p.team_1 = s.team_1
          AND p.team_2 = s.team_2
          AND p.map_name = s.map_name
          AND p.rank_1 = s.rank_1
          AND p.rank_2 = s.rank_2
          AND p.starting_ct = s.starting_ct
          AND p.map_winner = s.map_winner
          WHEN NOT MATCHED BY TARGET THEN
          INSERT *
          WHEN NOT MATCHED BY SOURCE THEN
          DELETE
          """,
    old=old_test_set,
    source=test_set,
)
logger.info("updated test set")

# Save to catalog
pandas_df = all_data.filter(F.col("map_winner").isNull()).toPandas()
outcomes = sample_outcomes(len(pandas_df), dists=dists)
to_update = spark.createDataFrame(pandas_df.assign(map_winner=outcomes))

# COMMAND ----------
spark.sql(
    """
          MERGE INTO {parsed} as p
          USING {source} as s
          ON p.date = s.date
          AND p.team_1 = s.team_1
          AND p.team_2 = s.team_2
          AND p.map_name = s.map_name
          AND p.rank_1 = s.rank_1
          AND p.rank_2 = s.rank_2
          AND p.starting_ct = s.starting_ct
          WHEN MATCHED THEN
            UPDATE SET
          p.map_winner = s.map_winner
          """,
    parsed=all_data,
    source=to_update,
)

logger.info("Attached Outcomes to matches")
