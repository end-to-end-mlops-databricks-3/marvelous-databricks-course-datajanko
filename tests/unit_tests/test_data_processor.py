"""Unit tests for DataProcessor."""

import datetime

import pandas as pd
import pytest

from mlops_course.config import SelectionConfig
from mlops_course.data_processor import basic_temporal_train_test_split, pre_processor


@pytest.mark.parametrize(
    "columns,config",
    [
        (
            {"date", "team_1", "map_winner"},
            SelectionConfig(features=["team_1"], date_column="date", target="map_winner"),
        ),
        (
            {"date", "team_1", "team_2", "result_1"},
            SelectionConfig(features=["team_1", "team_2"], date_column="date", target="result_1"),
        ),
    ],
)
def test_preprocessor(typed_data_sample: pd.DataFrame, config: SelectionConfig, columns: set[str]) -> None:
    """Test that the pre-processor selects the correct columns."""
    actual_columns = set(pre_processor(typed_data_sample, config).columns)
    assert actual_columns == columns


@pytest.mark.parametrize(
    "date_split,max_train,min_test",
    [
        (datetime.date(2020, 2, 28), pd.Timestamp("2020-02-27"), pd.Timestamp("2020-03-03")),
        (datetime.date(2020, 3, 9), pd.Timestamp("2020-03-09"), pd.Timestamp("2020-03-11")),
    ],
)
def test_split_data_default_params(
    typed_data_sample: pd.DataFrame, date_split: datetime.date, max_train: datetime.date, min_test: datetime.date
) -> None:
    """Test the default parameters of the split_data method in DataProcessor.

    This function tests if the split_data method correctly splits the input DataFrame
    into train and test
    """
    train, test = basic_temporal_train_test_split(typed_data_sample, date_split)
    assert train.date.max() == max_train
    assert test.date.min() == min_test
