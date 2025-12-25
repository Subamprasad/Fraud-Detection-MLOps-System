from src.models.rule_based import RuleBasedStrategy
from src.models.ml_model import MLStrategy
from src.config import Config

# --- FACTORY: Model Creation ---

class ModelFactory:
    @staticmethod
    def get_model(model_type: str):
        logger = Config().get_logger()
        logger.info(f"Factory creating model of type: {model_type}")
        
        if model_type == 'rule_based':
            return RuleBasedStrategy()
        elif model_type == 'ml':
            return MLStrategy()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
