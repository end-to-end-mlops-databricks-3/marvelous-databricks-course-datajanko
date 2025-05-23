import pandas as pd

from mlops_course.transformers import CategoryTransformer


def test_transformer() -> None:  # noqa
    df = pd.Series(["A", "B"])
    transformer = CategoryTransformer()
    transformer.fit(df)
    assert transformer._fitted
    print(transformer._categories)
    assert transformer._categories.to_list() == ["A", "B"]

    df2 = pd.Series(["A", "C", "B"])
    X = transformer.transform(df2)
    assert X.isnull().any().values[0]
