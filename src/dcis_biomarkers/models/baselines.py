from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any

class BaselineModels:
    """
    Fábrica de modelos baseline tradicionales para benchmarking comparativo.
    """
    @staticmethod
    def get_logistic_regression(random_state: int = 42, **kwargs) -> LogisticRegression:
        return LogisticRegression(class_weight="balanced", random_state=random_state, max_iter=1000, **kwargs)

    @staticmethod
    def get_random_forest(random_state: int = 42, **kwargs) -> RandomForestClassifier:
        return RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=random_state, **kwargs)
