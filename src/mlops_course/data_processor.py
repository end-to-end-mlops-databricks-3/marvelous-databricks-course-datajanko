"""Provding functions or classes to pre-process data before model training."""

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
