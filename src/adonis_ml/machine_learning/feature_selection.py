import logging 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, SelectFromModel, mutual_info_classif
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, method: str, params: dict, verbose=False):
        self.method = method
        self.params = params
        self.verbose = verbose

        self.selector = None

    def _build_selector(self):
        method = self.method
        params = self.params

        if method == "pca":
            return PCA(**params)
        elif method == "mutual_information":
            return SelectKBest(
                score_func=mutual_info_classif,
                **params
            )
        elif method == "lasso":
            return SelectFromModel(
                estimator=LogisticRegression(**params)
            )
        else:
            raise ValueError(f"Unknown feature selection method: {method}")

    def fit(self, X, y=None):
        if self.verbose:
            logger.info(f"Shape before feature selection (fit): {X.shape}")

        self.selector = self._build_selector()
        self.selector.fit(X, y)

        return self

    def transform(self, X):
        X_transformed = self.selector.transform(X)

        if self.verbose:
            logger.info(f"Shape after feature selection (transform): {X_transformed.shape}")
        return X_transformed

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)