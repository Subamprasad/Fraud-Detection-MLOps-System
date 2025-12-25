from typing import List, Protocol
import pandas as pd
from src.config import Config

# --- OBSERVER: Drift Monitoring ---

class DriftObserver(Protocol):
    def update(self, data: pd.DataFrame):
        pass

class DriftMonitor:
    def __init__(self):
        self._observers: List[DriftObserver] = []
        self.config = Config()
        self.logger = self.config.get_logger()
        self.baseline_stats = None
        self.drift_threshold = self.config.get('data', {}).get('drift_threshold', 0.05)

    def attach(self, observer: DriftObserver):
        self._observers.append(observer)

    def detach(self, observer: DriftObserver):
        self._observers.remove(observer)
    
    def set_baseline(self, data: pd.DataFrame):
        self.baseline_stats = data.mean(numeric_only=True)
        self.logger.info("Baseline stats set for drift monitoring.")

    def check_drift(self, new_data: pd.DataFrame):
        if self.baseline_stats is None:
            self.logger.warning("No baseline set. Skipping drift check.")
            return

        current_stats = new_data.mean(numeric_only=True)
        # Simple drift check: percent change in mean of features
        drift = (abs(current_stats - self.baseline_stats) / self.baseline_stats).max()
        
        self.logger.info(f"Max feature drift detected: {drift:.4f}")

        if drift > self.drift_threshold:
            self.logger.warning(f"Drift detected! Threshold: {self.drift_threshold}, Actual: {drift:.4f}")
            self.notify(new_data)

    def notify(self, data: pd.DataFrame):
        self.logger.info("Notifying observers of drift...")
        for observer in self._observers:
            observer.update(data)

class RetrainAlert(DriftObserver):
    def update(self, data: pd.DataFrame):
        logger = Config().get_logger()
        logger.info("[Observer Alert] Data drift significant. Recommendation: RETRAIN MODEL.")
