from abc import ABC, abstractmethod
import pandas as pd
from src.config import Config

# --- STRATEGY: Detection Strategies ---

class DetectionStrategy(ABC):
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series):
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        pass

class RuleBasedStrategy(DetectionStrategy):
    def __init__(self):
        self.config = Config()
        self.rules = self.config.get('models', {}).get('rule_based', {})
        self.logger = self.config.get_logger()

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.logger.info("Rule-based strategy does not require training.")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        self.logger.info("Running rule-based detection...")
        max_amount = self.rules.get('max_amount', 10000)
        suspicious_hours = self.rules.get('suspicious_hours', [])

        predictions = []
        for _, row in X.iterrows():
            is_fraud = 0
            if row.get('amount', 0) > max_amount:
                is_fraud = 1
            if row.get('time_of_day') in suspicious_hours:
                is_fraud = 1
            predictions.append(is_fraud)
        
        return pd.Series(predictions, index=X.index)
