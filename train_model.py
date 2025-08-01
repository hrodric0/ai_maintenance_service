import pandas as pd
import numpy as np
import joblib
import shap
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Step 1: Load synthetic dataset
def load_data(seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    return pd.DataFrame({
        "temperature": np.random.normal(75, 10, 1000),
        "vibration": np.random.normal(0.5, 0.1, 1000),
        "pressure": np.random.normal(30, 5, 1000),
        "runtime_hours": np.random.normal(500, 120, 1000),
        "failure": np.random.choice([0, 1], size=1000, p=[0.95, 0.05])
    })

# Step 2: Preprocess data
def preprocess(data: pd.DataFrame):
    X = data.drop("failure", axis=1)
    y = data["failure"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train model
def train_model(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Step 4: Evaluate model
def evaluate_model(model, X_test, y_test) -> None:
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)

# Step 5: Explain model with SHAP (optional)
def explain_model(model, X_sample: pd.DataFrame, show_plot: bool = False) -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    if show_plot:
        shap.summary_plot(shap_values[1], X_sample)

# Step 6: Save model artifact
def save_model(model, path: str = "model.pkl") -> None:
    joblib.dump(model, path)
    print(f"Model saved to {path}")

# Step 7: Track model with MLflow (optional)
def log_with_mlflow(model, X_test, y_test) -> None:
    mlflow.set_experiment("PredictiveMaintenance")
    with mlflow.start_run():
        acc = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
        print(f"Logged model to MLflow with accuracy: {acc:.4f}")

# Main training pipeline
if __name__ == "__main__":
    data = load_data()
    X_train, X_test, y_train, y_test = preprocess(data)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)
    log_with_mlflow(model, X_test, y_test)
    
    # Optional SHAP explanation
    explain_model(model, X_test.sample(min(100, len(X_test)), random_state=42), show_plot=True)

