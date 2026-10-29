import os
import urllib.request

import joblib
import matplotlib

# Use a non-interactive backend because this is a headless cloud server
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------
# Pooya Moradi
# St Code: 03221022705031
# Mohajer University © 2026 


DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/heart-disease/processed.cleveland.data"
)

os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

DATA_FILE = "data/processed.cleveland.data"
REPORT_FILE = "results/report.txt"
MODEL_FILE = "results/heart_disease_model.joblib"


# --------------------------------------------------
# 2. Download the dataset
# --------------------------------------------------

print("Downloading the Heart Disease dataset...")

try:
    request = urllib.request.Request(
        DATA_URL,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        dataset_bytes = response.read()

    with open(DATA_FILE, "wb") as file:
        file.write(dataset_bytes)

except Exception as error:
    print("Dataset download failed:", error)

    if not os.path.exists(DATA_FILE):
        raise SystemExit(
            "No local dataset was found. Check the server's internet connection."
        )

    print("Using the previously downloaded dataset.")


# --------------------------------------------------
# 3. Load and clean the data
# --------------------------------------------------

columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

df = pd.read_csv(
    DATA_FILE,
    header=None,
    names=columns,
    na_values="?",
)

print("Original dataset shape:", df.shape)

# Convert all columns to numeric values
df = df.apply(pd.to_numeric, errors="coerce")

# Remove rows containing missing values
df = df.dropna().copy()

# Convert the original target to a binary classification:
# 0 = no reported heart disease
# 1 = reported heart disease
df["target"] = (df["num"] > 0).astype(int)

# Remove the original multi-class target
df = df.drop(columns=["num"])

print("Clean dataset shape:", df.shape)
print("\nClass counts:")
print(df["target"].value_counts().sort_index())

# Save the cleaned dataset for inspection
df.to_csv("results/cleaned_heart_disease.csv", index=False)

X = df.drop(columns=["target"])
y = df["target"]


# --------------------------------------------------
# 4. Split into training and testing sets
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 5. Train the Random Forest model
# --------------------------------------------------

print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
)

model.fit(X_train, y_train)

print("Training completed.")


# --------------------------------------------------
# 6. Evaluate the model
# --------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["No reported disease", "Disease reported"],
    zero_division=0,
)

print("\n=== MODEL EVALUATION ===")
print("Accuracy: ", f"{accuracy:.4f}")
print("Precision:", f"{precision:.4f}")
print("Recall:   ", f"{recall:.4f}")
print("F1-score: ", f"{f1:.4f}")

print("\n=== CLASSIFICATION REPORT ===")
print(report)


# --------------------------------------------------
# 7. Save the evaluation report
# --------------------------------------------------

with open(REPORT_FILE, "w", encoding="utf-8") as file:
    file.write("HEART DISEASE CLASSIFICATION PROJECT\n")
    file.write("Algorithm: Random Forest\n")
    file.write("Dataset: UCI Cleveland Heart Disease\n\n")

    file.write(f"Original rows: {303}\n")
    file.write(f"Clean rows: {len(df)}\n")
    file.write(f"Training samples: {len(X_train)}\n")
    file.write(f"Testing samples: {len(X_test)}\n\n")

    file.write(f"Accuracy:  {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall:    {recall:.4f}\n")
    file.write(f"F1-score:  {f1:.4f}\n\n")

    file.write("CLASSIFICATION REPORT\n")
    file.write(report)

print("Evaluation report saved:", REPORT_FILE)


# --------------------------------------------------
# 8. Create a confusion matrix chart
# Compatible with scikit-learn 0.24.2
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

fig, ax = plt.subplots(figsize=(6, 5))

im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
fig.colorbar(im, ax=ax)

ax.set(
    xticks=[0, 1],
    yticks=[0, 1],
    xticklabels=["No disease", "Disease"],
    yticklabels=["No disease", "Disease"],
    xlabel="Predicted label",
    ylabel="Actual label",
    title="Heart Disease Classification",
)

# Print the number inside each cell
for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="black",
            fontsize=12,
        )

fig.tight_layout()
fig.savefig("results/confusion_matrix.png", dpi=150)
plt.close(fig)

print("Confusion matrix saved.")


# --------------------------------------------------
# 9. Plot feature importance
# --------------------------------------------------

importance = pd.Series(
    model.feature_importances_,
    index=X.columns,
).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 7))

importance.plot(kind="barh", ax=ax)

ax.set_title("Random Forest Feature Importance")
ax.set_xlabel("Importance")
ax.set_ylabel("Medical feature")

fig.tight_layout()
fig.savefig("results/feature_importance.png", dpi=150)
plt.close(fig)

print("Feature importance chart saved.")


# --------------------------------------------------
# 10. Save the trained model
# --------------------------------------------------

joblib.dump(model, MODEL_FILE)

print("Trained model saved:", MODEL_FILE)


# --------------------------------------------------
# 11. Final summary
# --------------------------------------------------

print("\n=== PROJECT COMPLETED SUCCESSFULLY ===")
print("Generated files:")

for filename in sorted(os.listdir("results")):
    path = os.path.join("results", filename)
    print(f" - {path}")

print("\nAll results are saved in the results/ directory.")