import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
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
# Define models and hyperparameters for Grid Search
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True)
}
params = {
    'Logistic Regression': {
        'classifier__C': [0.01, 0.1, 1]
    },
    'Random Forest': {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [None, 10, 20]
    },
    'SVM': {
        'classifier__C': [0.01, 0.1],
        'classifier__kernel': ['linear']
    }
}
# Perform Grid Search
best_model = None
best_params = None
best_score = 0

for model_name, model in models.items():
    # Create a pipeline with the preprocessor and the model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    # Perform Grid Search CV
    grid_search = GridSearchCV(pipeline, params[model_name], cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
   # Get the best model and score
    if grid_search.best_score_ > best_score:
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
# Save the best model with timestamp and accuracy in filename
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
accuracy = accuracy_score(y_test, best_model.predict(X_test))
model_filename = f'GS_model_{timestamp}_{accuracy:.4f}.joblib'
joblib.dump(best_model, model_filename)
# Predict on test data with the best model
y_pred = best_model.predict(X_test)
roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
# Print accuracy and ROC AUC score
print(f"Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")
# Save or append the Grid Search report
grid_search_file = 'GridSearch.md'
timestamp_heading = f"## {datetime.now().isoformat()}\n\n"
if os.path.exists(grid_search_file):
    with open(grid_search_file, 'a') as f:
        f.write(f"{timestamp_heading}")
        f.write(f"**Best Model:** {type(best_model.named_steps['classifier']).__name__}\n")
        f.write(f"**Best Hyperparameters:** {best_params}\n")
        f.write(f"**Best Score:** {best_score:.4f}\n")
else:
    with open(grid_search_file, 'w') as f:
        f.write(f"# Grid Search Results\n\n")
        f.write(f"{timestamp_heading}")
        f.write(f"**Best Model:** {type(best_model.named_steps['classifier']).__name__}\n")
        f.write(f"**Best Hyperparameters:** {best_params}\n")
        f.write(f"**Best Score:** {best_score:.4f}\n")
