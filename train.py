
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import pickle
import subprocess

BUCKET_NAME = "churn-mlops-bucket-caston"

# Load data from GCS
print("Loading data...")
train_df = pd.read_csv(f"gs://{BUCKET_NAME}/data/train.csv")
test_df = pd.read_csv(f"gs://{BUCKET_NAME}/data/test.csv")

X_train = train_df.drop(columns=["Churn"])
y_train = train_df["Churn"]
X_test = test_df.drop(columns=["Churn"])
y_test = test_df["Churn"]

print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# Train model
# class_weight='balanced' tells the model to pay extra attention to the
# minority class (churners) to compensate for the 73/27 imbalance
print("Training model...")
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
print("Evaluating model...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# Save model to GCS
print("Saving model...")
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

subprocess.run([
    "gsutil", "cp", "model.pkl",
    f"gs://{BUCKET_NAME}/models/model.pkl"
])
print("Model saved to GCS successfully")
