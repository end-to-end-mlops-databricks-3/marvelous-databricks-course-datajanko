import pandas as pd
import pytest

from mlops_course.transformers import CategoryTransformer


def test_category_transformer_happy_path() -> None:  # noqa
    df = pd.Series(["A", "B"])
    transformer = CategoryTransformer()
    transformer.fit(df)
    assert transformer._fitted
    print(transformer._categories)
    assert transformer._categories.to_list() == ["A", "B"]

    df2 = pd.Series(["A", "C", "B"])
    X = transformer.transform(df2)
    assert X.isnull().any().values[0]


def test_category_transformer_raised_on_dataframe() -> None:  # noqa
    df = pd.DataFrame(data={"col0": ["A", "B"], "col1": ["B", "C"]})
    transformer = CategoryTransformer()
    with pytest.raises(ValueError):
        transformer.fit(df)
