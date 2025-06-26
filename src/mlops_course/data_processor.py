"""Provding functions or classes to pre-process data before model training."""

import datetime
from datetime import timedelta

import pandas as pd
import pyspark.sql.functions as F
from loguru import logger
from marvelous.timer import Timer
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructField

from mlops_course.config import ProjectConfig, SelectionConfig

# 1 if not initial input exists
#   bootstrap (use pre_processor)
#
# don't want to pickle -> read max day from a config file -> this is just fro reproducability
# extract current max date from data
# generate matches for date +1 (append) to boostrapped table
# generate results for all matches not yet having a result and older than date+1 update table
# split now into train val and test, based on max_date
# update train set
# remove and attach from validation set
# remove and attach from test set


def pre_processor(df: pd.DataFrame, selection_config: SelectionConfig) -> pd.DataFrame:
    """Pre-processes the parsed data frame.

    Comments:
    - It's unclear whether we need to include the LabelEncoder already here.
    - TODO: more sophisticated feature computation
    """
    features = [selection_config.date_column] + selection_config.features + [selection_config.target]
    return df.filter(items=features)


def basic_temporal_train_test_split(
    df: pd.DataFrame, last_training_day: datetime.date
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a basic temporal train test split.

    We exploit that dates can be considered as normalized timestamps
    """
    return df.query("date <= @last_training_day"), df.query("date > @last_training_day")


def extract_distributions(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Extract distribution from cs go data columns."""
    # this is difficult to make config dependent, so we hard code it
    team_dist = df[["team_1", "team_2"]].stack().value_counts(normalize=True).rename_axis(index="teams")
    map_dist = df[["map_name"]].value_counts(normalize=True).rename_axis(index="map_name")
    starting_ct_dist = df[["starting_ct"]].value_counts(normalize=True).rename_axis(index="starting_ct")
    ranks_dist = df[["rank_1", "rank_2"]].stack().value_counts(normalize=True).rename_axis(index="rank")
    map_winners_dist = df[["map_winner"]].value_counts(normalize=True).rename_axis(index="map_winner")
    return {
        "team": team_dist,
        "map": map_dist,
        "starting_ct": starting_ct_dist,
        "ranks": ranks_dist,
        "winners": map_winners_dist,
    }


def sample_match(dists: dict[str, pd.Series]) -> pd.Series:
    """Sample single match from distirbutions."""
    teams = list(dists["team"].sample(2, weights=dists["team"]).index.tolist())
    map_name = [elem[0] for elem in dists["map"].sample(1, weights=dists["map"]).index.tolist()]
    starting_ct = [elem[0] for elem in dists["starting_ct"].sample(1, weights=dists["starting_ct"]).index.tolist()]
    ranks = list(dists["ranks"].sample(2, weights=dists["ranks"]).index.tolist())
    return pd.Series(
        teams + map_name + starting_ct + ranks,
        index=["team_1", "team_2", "map_name", "starting_ct", "rank_1", "rank_2"],
    )


def sample_matches(n: int, dists: dict[str, pd.Series]) -> pd.DataFrame:
    """Sample multiple matches."""
    matches = [sample_match(dists).to_frame().T for _ in range(n)]
    return pd.concat(matches, axis=0)


def sample_outcomes(n: int, dists: dict[str, pd.Series]) -> list[int]:
    """Sample outcomes of matches."""
    return [elem[0] for elem in dists["winners"].sample(n, replace=True).index.tolist()]


def drift_dists(dists: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """Introduce drift in distributions."""
    # we just switch the probabilities of a team, map and rank to depend on their lexicographical order
    drifted_teams = (
        dists["team"]
        .to_frame()
        .reset_index()
        .assign(teams=lambda df: df["teams"].sort_values().to_numpy())
        .set_index("teams")["proportion"]
    )
    drifted_map = (
        dists["map"]
        .to_frame()
        .reset_index()
        .assign(map_name=lambda df: df["map_name"].sort_values().to_numpy())
        .set_index("map_name")["proportion"]
    )
    drifted_ranks = (
        dists["ranks"]
        .to_frame()
        .reset_index()
        .assign(map_name=lambda df: df["rank"].sort_values().to_numpy())
        .set_index("rank")["proportion"]
    )

    # Change probabilities of starting cts and winning classes
    drifted_starting_ct = dists["starting_ct"].copy()
    drifted_starting_ct.update(pd.Series([0.8, 0.2], index=[1, 2]))

    drifted_winners = dists["winners"].copy()
    drifted_winners.update(pd.Series([0.65, 0.35], index=[1, 2]))

    return {
        "team": drifted_teams,
        "map": drifted_map,
        "starting_ct": drifted_starting_ct,
        "ranks": drifted_ranks,
        "winners": drifted_winners,
    }


def bootstrap(config: ProjectConfig, is_bootstrap: int, max_date: datetime.date, spark: SparkSession) -> None:
    """Bootstraps parsed, training, validation and test data."""
    bootstrap_necessary = spark.catalog.tableExists(f"{config.catalog_name}.{config.schema_name}.parsed_data")
    val_offset = config.validation_size_in_days
    test_offset = config.test_set_size_in_days

    if is_bootstrap == 1 or bootstrap_necessary:
        logger.info("Start Populating data sources")

        with Timer() as bootstrap_timer:
            df = spark.read.csv(
                f"/Volumes/{config.catalog_name}/{config.schema_name}/data/results.csv", header=True, inferSchema=True
            )
            processed_data = (
                df.withColumn(config.selection.date_column, F.col(config.selection.date_column).cast("timestamp"))
                .withColumnsRenamed(config.parsing.rename)
                .select([config.selection.date_column] + config.selection.features + [config.selection.target])
            )  # type:ignore

            processed_data.write.mode("overwrite").format("delta").option("overwriteSchema", True).saveAsTable(
                f"{config.catalog_name}.{config.schema_name}.parsed_data"
            )
            # Need to check utc issues
            test_end = datetime.datetime(year=max_date.year, month=max_date.month, day=max_date.day)

            validation_end = test_end - timedelta(days=test_offset)  # noqa # type: ignore
            training_end = validation_end - timedelta(days=val_offset)  # noqa # type: ignore

            train_set = processed_data.filter(F.col("date") <= F.lit(training_end))
            validation_set = processed_data.filter(F.col("date") > F.lit(training_end)).filter(
                F.col("date") <= F.lit(validation_end)
            )
            test_set = processed_data.filter(F.col("date") > F.lit(validation_end)).filter(
                F.col("date") <= F.lit(test_end)
            )

            train_set.write.mode("overwrite").format("delta").option("overwriteSchema", True).saveAsTable(
                f"{config.catalog_name}.{config.schema_name}.train_set"
            )
            validation_set.write.mode("overwrite").format("delta").option("overwriteSchema", True).saveAsTable(
                f"{config.catalog_name}.{config.schema_name}.validation_set"
            )
            test_set.write.mode("overwrite").format("delta").option("overwriteSchema", True).saveAsTable(
                f"{config.catalog_name}.{config.schema_name}.test_set"
            )

        logger.info(f"Bootstrapping Completed! Took: {bootstrap_timer}")


def attach_generated_data(
    config: ProjectConfig,
    max_date: datetime.date,
    spark: SparkSession,
    schema: StructField,
    original_data: DataFrame,
    drift: bool,
) -> dict[str, pd.Series]:
    """Attach newly generated matches without and outcome."""
    dists = extract_distributions(original_data.drop("date").toPandas())
    if drift == 1:
        dists = drift_dists(dists)
    sampled_matches = sample_matches(10, dists).assign(map_winner=None, date=max_date + timedelta(days=1))[
        [config.selection.date_column] + config.selection.features + [config.selection.target]
    ]
    sampled_matches_with_date = spark.createDataFrame(sampled_matches, schema=schema)

    sampled_matches_with_date.write.mode("append").format("delta").saveAsTable(
        f"{config.catalog_name}.{config.schema_name}.parsed_data"
    )
    logger.info("Synthetic matches generated and attached")

    return dists


def create_data_split_and_update(
    config: ProjectConfig, spark: SparkSession, val_offset: int, test_offset: int, all_data: DataFrame
) -> None:
    """Update train, test and validation split."""
    max_date_with_results = all_data.filter(F.col("map_winner").isNotNull()).agg({"date": "max"}).collect()[0][0]

    old_train_set = spark.read.table(f"{config.catalog_name}.{config.schema_name}.train_set")
    old_valid_set = spark.read.table(f"{config.catalog_name}.{config.schema_name}.validation_set")
    old_test_set = spark.read.table(f"{config.catalog_name}.{config.schema_name}.test_set")

    test_end = max_date_with_results
    validation_end = test_end - timedelta(days=test_offset)  # type:ignore
    training_end = validation_end - timedelta(days=val_offset)  # type: ignore

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
def observe_outcomes(spark: SparkSession, all_data: DataFrame, dists: dict[str, pd.Series]) -> None:
    """Update parsed data with outcomes of matches."""
    pandas_df = all_data.filter(F.col("map_winner").isNull()).toPandas()
    outcomes = sample_outcomes(len(pandas_df), dists=dists)
    to_update = spark.createDataFrame(pandas_df.assign(map_winner=outcomes))  # type: ignore

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
