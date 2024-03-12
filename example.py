import mlflow
from mlflow.models import infer_signature
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

### 1) START A TRACKING SERVER
# Setup local tracking server
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")


### 2) TRAIN A MODEL AND PREPARE METDATA FOR LOGGING
# Load the Iris dataset
X, y = datasets.make_classification(n_samples=1000)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model hyperparameters
params = {
    "solver": "liblinear",
    "max_iter": 50,
    "multi_class": "auto",
    "random_state": 8888,
}

# Train the model
log_reg = LogisticRegression(**params)
log_reg.fit(X_train, y_train)

# Predict on the test set
y_pred = log_reg.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
print("accuracy: {0}".format(accuracy))


### 3) LOG THE MODEL AND ITS METADATA
# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("Classification Test")

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the loss metric
    mlflow.log_metric("accuracy", accuracy)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic LogReg model for iris data")

    # Infer the model signature
    signature = infer_signature(X_train, log_reg.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=log_reg,
        artifact_path="make_classfication",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-quickstart",
    )