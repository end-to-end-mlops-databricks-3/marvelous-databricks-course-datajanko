"""Provding functions or classes to pre-process data before model training."""

import datetime

import pandas as pd

from mlops_course.config import SelectionConfig


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
