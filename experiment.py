import os
import tempfile
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datetime import datetime

# Set up a temporary directory for MLflow artifacts
artifact_location = tempfile.mkdtemp()

# Configure MLflow to use the temporary directory for artifacts
mlflow.set_tracking_uri(f'file://{artifact_location}')

# Load the data
data = pd.read_csv('diabetes_dataset.csv')

# Define columns
categorical_cols = ['gender', 'location', 'smoking_history']
numerical_cols = ['year', 'age', 'bmi', 'hbA1c_level', 'blood_glucose_level']

# Define preprocessing for numerical columns (scale them)
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

# Define preprocessing for categorical columns (encode them)
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Create a preprocessor with ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

# Prepare target variable and features
X = data.drop('diabetes', axis=1)
y = data['diabetes']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply transformations
X_train_prepared = preprocessor.fit_transform(X_train)
X_test_prepared = preprocessor.transform(X_test)                  

# Set up MLflow experiment
mlflow.set_experiment("diabetes_classification")

# Experiment with different hyperparameters and models
models = {
    "LogisticRegression": {
        "model": LogisticRegression,
        "params": [
            {"max_iter": 1000, "C": 1.0},
            {"max_iter": 1000, "C": 0.5},
            {"max_iter": 500, "C": 1.0}
        ]
    },
    "RandomForestClassifier": {
        "model": RandomForestClassifier,
        "params": [
            {"n_estimators": 100, "max_depth": None},
            {"n_estimators": 200, "max_depth": 10},
            {"n_estimators": 100, "max_depth": 5}
        ]
    }
}

best_accuracy = 0
best_model = None
best_model_name = ""
best_params = {}

for model_name, model_info in models.items():
    ModelClass = model_info["model"]
    for params in model_info["params"]:
        with mlflow.start_run():
            # Log parameters
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            mlflow.log_param("model_type", model_name)
            
            # Initialize and train the model
            model = ModelClass(**params)
            model.fit(X_train_prepared, y_train)

            # Predict on test data
            y_pred = model.predict(X_test_prepared)
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_prepared)[:, 1])

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("roc_auc", roc_auc)

            # Log the model
            mlflow.sklearn.log_model(model, "model")

            # Print the results
            print(f"Model: {model_name}, Params: {params}, Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")

            # Log the preprocessing pipeline
            joblib.dump(preprocessor, "preprocessor.pkl")
            mlflow.log_artifact("preprocessor.pkl")

            # Update best model if current one is better
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_name = model_name
                best_params = params

                # Save best model information
                with open("best_model_name.txt", "w") as f:
                    f.write(best_model_name)
                with open("best_model_params.txt", "w") as f:
                    f.write(str(best_params))
                with open("best_model_accuracy.txt", "w") as f:
                    f.write(str(best_accuracy))

# Save the best model with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
best_model_filename = f"{timestamp}_best_model.joblib"
joblib.dump(best_model, best_model_filename)

# Print best model and parameters
print(f"Best Model: {best_model_name}")
print(f"Best Hyperparameters: {best_params}")
print(f"Best Accuracy: {best_accuracy:.4f}")

# Save the best model and preprocessing pipeline together
joblib.dump({
    "model": best_model,
    "preprocessor": preprocessor
}, best_model_filename)
