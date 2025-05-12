import pandas as pd

from mlops_course.config import ParsingConfig


def csv_data_loader(file_path: str, columns: list[str], config: ParsingConfig) -> pd.DataFrame:
    """Read columns from csv file given a file_path.

    Parse the contents according to a ParsingConfig.
    """
    df = pd.read_csv(file_path).filter(items=columns).rename(config.rename, axis=1)
    for col in config.categories:
        df[col] = df[col].astype("category")
    df["date"] = pd.to_datetime(df[config.date_column], erros="coerce")  # type: ignore
    return df
