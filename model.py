import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datetime import datetime
import os

# Load dataset
data = pd.read_csv('diabetes_dataset.csv')

# Define columns
categorical_cols = ['gender', 'location', 'smoking_history']
numerical_cols = ['year', 'age', 'bmi', 'hbA1c_level', 'blood_glucose_level']

#To demonstrate version control
print(categorical_cols)

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

# Create the final pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', LogisticRegression(max_iter=1000))])

# Fit the model
pipeline.fit(X_train, y_train)

# Generate timestamp for filenames
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

# Save the model and preprocessing steps with timestamp
model_filename = f'model_{timestamp}.joblib'
joblib.dump(pipeline, model_filename)

# Predict on test data
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])

# Print accuracy and ROC AUC score
print(f"Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")

# Save or append the report
report_file = 'report.md'
timestamp_heading = f"## Report {datetime.now().isoformat()}\n\n"

if os.path.exists(report_file):
    with open(report_file, 'a') as f:
        f.write(f"{timestamp_heading}")
        f.write(f"**Accuracy:** {accuracy:.4f}\n\n")
        f.write(f"**ROC AUC:** {roc_auc:.4f}\n")
else:
    with open(report_file, 'w') as f:
        f.write(f"# Model Report\n\n")
        f.write(f"{timestamp_heading}")
        f.write(f"**Accuracy:** {accuracy:.4f}\n\n")
        f.write(f"**ROC AUC:** {roc_auc:.4f}\n")
