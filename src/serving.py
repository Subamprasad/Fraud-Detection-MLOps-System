import time
from src.config import Config
from src.models.rule_based import DetectionStrategy

# --- DECORATOR: Logging and Metrics ---

class PredictionService:
    def __init__(self, strategy: DetectionStrategy):
        self.strategy = strategy

    def predict(self, data):
        return self.strategy.predict(data)

class LoggingDecorator:
    def __init__(self, service: PredictionService):
        self.service = service
        self.logger = Config().get_logger()

    def predict(self, data):
        start_time = time.time()
        self.logger.info(f"Analysis Request: {len(data)} transactions")
        
        result = self.service.predict(data)
        
        duration = time.time() - start_time
        fraud_count = result.sum()
        self.logger.info(f"Analysis Complete. Duration: {duration:.4f}s. Fraud Detected: {fraud_count}")
        return result

# --- PROXY: Access Control ---

class ModelServingProxy:
    def __init__(self, service: PredictionService, role: str):
        self.service = service
        self.role = role
        self.logger = Config().get_logger()

    def predict(self, data):
        if self.role != "ADMIN" and self.role != "SYSTEM":
            self.logger.error(f"Access Denied: Role '{self.role}' is not authorized to request predictions.")
            raise PermissionError("Access Denied")
        
        self.logger.info(f"Access Granted for role: {self.role}")
        return self.service.predict(data)
