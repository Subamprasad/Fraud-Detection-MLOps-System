# Fraud Detection MLOps System

A production-oriented MLOps system demonstrating the evolution from rule-based logic to machine learning, built with robust software design patterns.

## 🚀 Features

- **Multi-Strategy Detection**: Seamlessly switch between Rule-Based and ML (Logistic Regression) strategies.
- **Drift Monitoring**: Automated drift detection on feature distributions triggers alerts.
- **Secure Serving**: Proxy pattern ensures only authorized roles can access predictions.
- **Observability**: Decorators provide automatic logging and metrics for every prediction.
- **Feedback Loop**: Command pattern simplifies orchestrating training and retraining pipelines.

## 🏗 Design Patterns Used

1. **Singleton**: `src/config.py` - Ensures a single source of truth for config and logging.
2. **Adapter**: `src/config.py` - Adapts data sources to a standard DataFrame format.
3. **Observer**: `src/drift_monitor.py` - Monitors data streams and notifies the system of drift.
4. **Strategy**: `src/models/` - Interchangeable detection algorithms (RuleBased vs ML).
5. **Factory**: `src/models/factory.py` - Centralized creation of model instances.
6. **Command**: `src/commands.py` - Encapsulates Train/Predict/Retrain actions as objects.
7. **Decorator**: `src/serving.py` - Adds logging/metrics without executing core logic.
8. **Proxy**: `src/serving.py` - Controls access to the model serving layer.

## 📂 Structure
```
fraud_detection_system/
├── config/             # Configuration files
├── src/
│   ├── models/         # Strategy & Factory implementations
│   ├── commands.py     # Command pattern
│   ├── config.py       # Singleton & Adapter
│   ├── drift_monitor.py# Observer pattern
│   ├── serving.py      # Proxy & Decorator
└── main.py             # Entry point
```

## 🏃 parameters to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the System**:
   ```bash
   python main.py
   ```

## 🔄 Workflow Explained

1. **Initialization**: Config loads, Logger starts.
2. **Baseline**: System starts with a Rule-Based model for immediate value.
3. **Training**: ML model is trained in the background (via Command).
4. **Serving**: Requests are proxied and decorated for safety and visibility.
5. **Monitoring**: If input data drifts (simulated in `main.py`), an Observer alerts the system, triggering a Retrain Command.

## 🛡 Design Pattern Validation and MLOps Reasoning

**Build → Validate → Explain**
This system was built with a strict "Design First" mindset. In MLOps, components like data ingestion, model training, and serving have distinct lifecycles and requirements. We use standard software design patterns to decouple these components, ensuring that the system is robust, testable, and capable of evolving (e.g., swapping a model or data source) without breaking the entire pipeline.

All 8 design patterns (Singleton, Factory, Observer, Adapter, Strategy, Command, Decorator, Proxy) have been implemented and validated to ensure a robust and scalable architecture.

### 🧠 MLOps Decisions Explained

**Why start with a Rule-Based Baseline?**
Machine learning is not always the first answer. A rule-based baseline provides immediate business value (blocking large transactions at night) and sets a performance benchmark. We only add ML (Logistic Regression) when we have a pipeline to compare it against the baseline.

**Why Separate Training and Serving?**
The `TrainCommand` happens offline (batch), while `PredictionService` (Proxy/Decorator) handles online inference. This separation allows us to scale them independently—training can be resource-intensive and infrequent, while serving must be lightweight and low-latency.

**How the Feedback Loop Works**
The **Observer** pattern closes the loop. Instead of manually checking reports, the `DriftMonitor` subscribes to the data stream. When `drift > threshold`, it triggers the `RetrainCommand` programmatically, automating the maintenance of model performance.
