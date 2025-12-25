import pandas as pd
from src.config import Config, CSVAdapter
from src.models.factory import ModelFactory
from src.commands import TrainCommand, RetrainCommand
from src.serving import PredictionService, LoggingDecorator, ModelServingProxy
from src.drift_monitor import DriftMonitor, RetrainAlert

def main():
    # 1. Initialize System
    config = Config() # Singleton
    logger = config.get_logger()
    logger.info("=== FRAUD DETECTION MLOps SYSTEM STARTING ===")

    # 2. Load Data (Adapter)
    data_adapter = CSVAdapter()
    data = data_adapter.load_data()
    
    # Split features and target
    X = data.drop(columns=['is_fraud'])
    y = data['is_fraud']

    # 3. Setup Drift Monitoring (Observer)
    monitor = DriftMonitor()
    alert = RetrainAlert()
    monitor.attach(alert)
    monitor.set_baseline(X.iloc[:500]) # First 500 samples as baseline

    # 4. Phase 1: Rule-Based Baseline (Strategy)
    logger.info("\n--- PHASE 1: Rule-Based Baseline ---")
    rule_model = ModelFactory.get_model('rule_based')
    
    # Use Proxy and Decorator for serving
    service = PredictionService(rule_model)
    decorated_service = LoggingDecorator(service)
    proxy = ModelServingProxy(decorated_service, role="SYSTEM")
    
    predictions = proxy.predict(X.iloc[500:510]) # Predict on small batch
    logger.info(f"Baseline Predictions: {predictions.values}")

    # 5. Phase 2: ML Model Training (Command)
    logger.info("\n--- PHASE 2: ML Model Training ---")
    ml_model = ModelFactory.get_model('ml')
    train_cmd = TrainCommand(ml_model, X.iloc[:800], y.iloc[:800])
    train_cmd.execute()

    # 6. Phase 3: Serving ML Model (Proxy -> Decorator -> Service)
    logger.info("\n--- PHASE 3: ML Model Serving ---")
    ml_service = PredictionService(ml_model)
    ml_decorated = LoggingDecorator(ml_service)
    ml_proxy = ModelServingProxy(ml_decorated, role="SYSTEM")
    
    # Simulate batch prediction
    batch_data = X.iloc[800:850]
    ml_predictions = ml_proxy.predict(batch_data)
    
    # 7. Phase 4: Drift Detection & Automated Retraining loop
    logger.info("\n--- PHASE 4: Monitoring & Feedback Loop ---")
    
    # Simulate drift by multiplying input data features
    drifted_data = batch_data.copy() * 1.5 
    monitor.check_drift(drifted_data)
    
    # In a real system, the Observer would trigger this, here we simulate the flow
    logger.info("Triggering automated retraining based on drift...")
    retrain_cmd = RetrainCommand(ml_model, data_adapter)
    retrain_cmd.execute()

    logger.info("\n=== SYSTEM EXECUTION COMPLETE ===")

if __name__ == "__main__":
    main()
