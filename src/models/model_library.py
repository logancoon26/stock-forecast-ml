from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from src.models.HybridCNNModel import HybridCNNModel
from src.models.TransformerModel import StockTransformer


def get_model(model_name, random_state=42, **kwargs):
    """
    Returns benchmark and custom ML models.
    """

    if model_name == "logistic_regression":
        return LogisticRegression(
        max_iter=200,
        random_state=random_state
    )

    elif model_name == "random_forest":
        return RandomForestClassifier(
        n_estimators=100,
        random_state=random_state
    )

    elif model_name == "svm":
        return CalibratedClassifierCV(
            LinearSVC(random_state=random_state, max_iter=2000)
        )
    
    elif model_name == "xgboost":
        return XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=random_state,
            eval_metric="logloss",
        )

    elif model_name == "hybrid":
        return HybridCNNModel(
        **kwargs,
    )

    elif model_name == "transformer":
        return StockTransformer(
            **kwargs
        )

    else:
        raise ValueError(f"Model {model_name} not recognized.")  