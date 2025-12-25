import yaml
import logging
import os
import pandas as pd
from typing import List, Dict, Any

# --- SINGLETON: Configuration and Logging ---
class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
            cls._instance._setup_logging()
        return cls._instance

    def _load_config(self):
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')
        with open(config_path, 'r') as f:
            self.settings = yaml.safe_load(f)

    def _setup_logging(self):
        log_level = self.settings['app']['log_level']
        logging.basicConfig(level=getattr(logging, log_level), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(self.settings['app']['name'])

    def get(self, key: str, default=None) -> Any:
        return self.settings.get(key, default)
    
    def get_logger(self):
        return self.logger

# --- ADAPTER: Data Ingestion ---
class DataSourceAdapter:
    """
    Target interface that the system expects.
    """
    def load_data(self) -> pd.DataFrame:
        pass

class CSVAdapter(DataSourceAdapter):
    """
    Adaptee: Loads data from CSV (or simulates it for this demo).
    """
    def __init__(self, file_path: str = None):
        self.file_path = file_path
        self.logger = Config().get_logger()

    def load_data(self) -> pd.DataFrame:
        if self.file_path and os.path.exists(self.file_path):
            self.logger.info(f"Loading data from {self.file_path}")
            return pd.read_csv(self.file_path)
        else:
            self.logger.warning("File not found or provided. Generating dummy data.")
            return self._generate_dummy_data()

    def _generate_dummy_data(self) -> pd.DataFrame:
        import numpy as np
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'amount': np.random.exponential(scale=100, size=n_samples),
            'time_of_day': np.random.randint(0, 24, size=n_samples),
            'merchant_risk': np.random.uniform(0, 1, size=n_samples),
            'location_distance': np.random.exponential(scale=5, size=n_samples),
            'is_fraud': [0] * n_samples # Placeholder
        }
        
        df = pd.DataFrame(data)
        
        # Inject fraud based on rules to make it realistic
        def simulate_label(row):
            score = 0
            if row['amount'] > 500: score += 5
            if row['time_of_day'] < 4: score += 3
            if row['merchant_risk'] > 0.8: score += 4
            return 1 if score > 5 else 0

        df['is_fraud'] = df.apply(simulate_label, axis=1)
        return df
