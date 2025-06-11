# Pipeline class for data preprocessing
# Pipeline, allows us to define our steps in a certain order.
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from .load_config import PreprocessorConfig, load_config

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, config: PreprocessorConfig):
        # If transform is run without fitting, it will show an error here.
        self._is_fitted = False
        self.config = config

    # Column Transformer: Hangi kolonlara hangi transformer uygulayacağı kolaylaştıran bir scikit-learn class'ıdır.
    def __build_column_transformer(self):
        # (name, trans, col)
        transformers = []
        for feature in self.config.features.numerical:
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append((feature, numerical_pipeline, [feature]))

        for feature in self.config.features.categorical:
            categorical_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            )
            transformers.append((feature, categorical_pipeline, [feature]))
        return ColumnTransformer(transformers=transformers, remainder="drop")

    def fit(self, X, y=None):
        self.pipeline = self.__build_column_transformer()
        self.pipeline.fit(X, y)
        self._is_fitted = True
        return self

    def transform(self, X):
        if not self._is_fitted:
            raise RuntimeError(
                "You must fit the DataPreprocessor before calling transform"
            )
        return self.pipeline.transform(X)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
