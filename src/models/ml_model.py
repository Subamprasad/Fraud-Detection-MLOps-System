import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.config import Config
from src.models.rule_based import DetectionStrategy # Importing Interface

class MLStrategy(DetectionStrategy):
    def __init__(self):
        self.config = Config()
        self.logger = self.config.get_logger()
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression())
        ])
        self.is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.logger.info("Training ML model (Logistic Regression)...")
        self.model.fit(X, y)
        self.is_trained = True
        self.logger.info("Training complete.")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self.is_trained:
            self.logger.warning("Model not trained! Returning zeros.")
            return pd.Series([0]*len(X), index=X.index)
        
        self.logger.info("Running ML-based detection...")
        return pd.Series(self.model.predict(X), index=X.index)
