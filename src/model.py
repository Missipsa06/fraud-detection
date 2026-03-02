from lightgbm import LGBMClassifier
from .config import MODEL_PARAMS

def train_model(X_train, y_train, params=None):
    model = LGBMClassifier(**(params or MODEL_PARAMS))
    model.fit(X_train, y_train)
    return model

def predict_proba(model, X):
    return model.predict_proba(X)[:, 1]