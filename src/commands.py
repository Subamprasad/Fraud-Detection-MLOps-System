from abc import ABC, abstractmethod
from src.config import Config
from src.models.rule_based import DetectionStrategy

# --- COMMAND: Operational Actions ---

class Command(ABC):
    @abstractmethod
    def execute(self):
        pass

class TrainCommand(Command):
    def __init__(self, model: DetectionStrategy, train_data, train_labels):
        self.model = model
        self.train_data = train_data
        self.train_labels = train_labels
        self.logger = Config().get_logger()

    def execute(self):
        self.logger.info("Executing TrainCommand...")
        self.model.train(self.train_data, self.train_labels)

class PredictCommand(Command):
    def __init__(self, model: DetectionStrategy, data):
        self.model = model
        self.data = data
        self.logger = Config().get_logger()
        self.result = None

    def execute(self):
        self.logger.info("Executing PredictCommand...")
        self.result = self.model.predict(self.data)
        return self.result

class RetrainCommand(Command):
    def __init__(self, model: DetectionStrategy, data_loader):
        self.model = model
        self.data_loader = data_loader
        self.logger = Config().get_logger()

    def execute(self):
        self.logger.info("Executing RetrainCommand...")
        fresh_data = self.data_loader.load_data()
        X = fresh_data.drop(columns=['is_fraud'])
        y = fresh_data['is_fraud']
        
        subset = int(len(X) * 0.8) # Simple retrain logic
        self.model.train(X.iloc[:subset], y.iloc[:subset])
        self.logger.info("Retraining complete.")
