import mlflow
import mlflow.onnx
import onnx
import onnxruntime as ort

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# -----------------------------
# Configuration
# -----------------------------

MODEL_PATH = "mnist-12.onnx"

BATCH_SIZE = 128

DEVICE = "CPU"

# -----------------------------
# Dataset
# -----------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
])

test_dataset = datasets.EMNIST(
    root="./data",
    split="digits",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# -----------------------------
# ONNX Runtime
# -----------------------------

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

# -----------------------------
# MLflow
# -----------------------------

mlflow.set_experiment("Digit Recognition Evaluation")

with mlflow.start_run():

    # Parameters
    mlflow.log_param("Framework", "ONNX Runtime")
    mlflow.log_param("Dataset", "EMNIST Digits")
    mlflow.log_param("Batch Size", BATCH_SIZE)
    mlflow.log_param("Device", DEVICE)

    y_true = []
    y_pred = []

    # -------------------------
    # Inference
    # -------------------------

    for images, labels in test_loader:
        batch_preds = []
        for i in range(images.shape[0]):
            # Extract a single image and add batch dimension -> [1, 1, 28, 28]
            single_img = np.expand_dims(images[i].numpy(), axis=0)
            outputs = session.run(
                None,
                {
                    input_name: single_img
                }
            )[0]
            pred = np.argmax(outputs, axis=1)[0]
            batch_preds.append(pred)

        y_true.extend(labels.numpy())
        y_pred.extend(batch_preds)

    # -------------------------
    # Metrics
    # -------------------------

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted"
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted"
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted"
    )

    print("\nEvaluation Results\n")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # Log metrics

    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1 Score", f1)

    # -------------------------
    # Confusion Matrix
    # -------------------------

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8,8))

    disp = ConfusionMatrixDisplay(cm)

    disp.plot(cmap="Blues", colorbar=False)

    plt.title("Confusion Matrix")

    plt.savefig("confusion_matrix.png")

    mlflow.log_artifact("confusion_matrix.png")

    plt.close()

    # -------------------------
    # Classification Report
    # -------------------------

    report = classification_report(y_true, y_pred)

    print(report)

    with open("classification_report.txt","w") as f:
        f.write(report)

    mlflow.log_artifact("classification_report.txt")

    # -------------------------
    # Log ONNX model
    # -------------------------

    onnx_model = onnx.load(MODEL_PATH)

    mlflow.onnx.log_model(
        onnx_model,
        artifact_path="model"
    )

print("\nFinished Successfully.")