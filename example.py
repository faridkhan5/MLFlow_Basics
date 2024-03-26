import mlflow
import os
import sys
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score



### 1) START A TRACKING SERVER
# Set our tracking server uri for logging

## For local host only
#mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

## for remote server only (Dagshub)
# remote_server_uri = 'https://dagshub.com/faridkhan5/MLFlow_Basics.mlflow'
# mlflow.set_tracking_uri(remote_server_uri)

## for remote server only (AWS)
remote_server_uri = 'http://ec2-16-170-133-156.eu-north-1.compute.amazonaws.com:5000/'
mlflow.set_tracking_uri(remote_server_uri)

### 2) TRAIN A MODEL AND PREPARE METDATA FOR LOGGING
# Load the Iris dataset
X, y = datasets.make_classification(n_samples=1000)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model hyperparameters
params = {
    'penalty' : sys.argv[1] if len(sys.argv) > 1 else 'l2',
    'solver' : sys.argv[2] if len(sys.argv) > 2 else 'lbfgs',
    'C' : float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
}


# Train the model
log_reg = LogisticRegression(**params)

log_reg.fit(X_train, y_train)

# Predict on the test set
y_pred = log_reg.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"accuracy: {accuracy:.3f}")
print(f"precision: {precision:.3f}")
print(f"recall: {recall:.3f}")


### 3) LOG THE MODEL AND ITS METADATA
# Create a new MLflow Experiment
#terminal -> `mlflow experiments create --experiment-name [name]`
mlflow.set_experiment(experiment_name="make_classification_dataset")

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the loss metric
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic LogReg model for make_classification data")

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=log_reg,
        artifact_path="make_classfication",
        input_example=X_train,
        registered_model_name="Classification_LogReg_Model",
    )