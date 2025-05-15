from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession

from mlops_course.config import ParsingConfig, ProjectConfig


def csv_data_loader(file_path: str | Path, columns: list[str], config: ParsingConfig) -> pd.DataFrame:
    """Read columns from csv file given a file_path.

    Parse the contents according to a ParsingConfig.
    """
    df = pd.read_csv(file_path).filter(items=columns).rename(config.rename, axis=1)
    for col in config.categories:
        df[col] = df[col].astype("category")
    df["date"] = pd.to_datetime(df[config.date_column], errors="coerce")  # type: ignore
    return df


def store_processed(processed: pd.DataFrame, config: ProjectConfig, spark: SparkSession) -> None:
    """Store processed data frame for later feature computation in unity catalog.

    Override data to ensure that jobs don't complain (for now)
    """
    spark.createDataFrame(processed).write.mode("overwrite").option("overwriteSchema", True).saveAsTable(
        f"{config.catalog_name}.{config.schema_name}.parsed_data"
    )


def store_train_test(train: pd.DataFrame, test: pd.DataFrame, config: ProjectConfig, spark: SparkSession) -> None:
    """Store Train and test sets in unity catalog.

    Ignores the timestamp treatment from the parent repo for now
    """
    spark.createDataFrame(train).write.mode("overwrite").saveAsTable(
        f"{config.catalog_name}.{config.schema_name}.train_set"
    )

    spark.createDataFrame(test).write.mode("overwrite").saveAsTable(
        f"{config.catalog_name}.{config.schema_name}.test_set"
    )
