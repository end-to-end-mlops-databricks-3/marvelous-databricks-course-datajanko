from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from mlops_course.config import ProjectConfig
from mlops_course.data_io import csv_data_loader


def test_csv_loader(raw_data_file_path: Path, typed_data_sample: pd.DataFrame, config: ProjectConfig) -> None:
    """Ensures data loading provides expected outputs.

    We verify that the csv loader functions provides expected outputs for a given configuration
    """
    given_file_path = raw_data_file_path
    given_columns = config.raw_data_columns
    given_parsing_config = config.parsing
    actual = csv_data_loader(given_file_path, given_columns, given_parsing_config)
    expected = typed_data_sample
    assert_frame_equal(actual, expected)
