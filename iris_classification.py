"""
QSkill Internship Project 1: Iris Flower Classification
Run: python iris_classification.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, auc
)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("=" * 60)
    print("QSKILL INTERNSHIP: IRIS FLOWER CLASSIFICATION")
    print("=" * 60)

    # 1. Load the built-in Iris dataset (no separate download required).
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    df = X.copy()
    df["species"] = y.map(dict(enumerate(target_names)))

    print("\nFirst five rows:")
    print(df.head().to_string(index=False))
    print("\nDataset shape:", df.shape)
    print("\nSpecies counts:")
    print(df["species"].value_counts().to_string())
    print("\nMissing values:", int(df.isnull().sum().sum()))

    # 2. Explore the data visually.
    sns.pairplot(df, hue="species", diag_kind="hist")
    plt.suptitle("Iris Dataset: Feature Relationships", y=1.02)
    plt.savefig(os.path.join(OUTPUT_DIR, "iris_pairplot.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.heatmap(X.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Between Iris Features")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_correlation.png"), dpi=150)
    plt.close()

    # 3. Split data into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 4. Scale numeric features and train a classifier.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    model.fit(X_train, y_train)

    # 5. Evaluate the model.
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Iris Classification - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # 6. Multiclass ROC curves and AUC using one-vs-rest probabilities.
    y_test_bin = label_binarize(y_test, classes=np.arange(len(target_names)))
    y_score = model.predict_proba(X_test)

    plt.figure(figsize=(8, 6))
    for i, name in enumerate(target_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        print(f"ROC-AUC for {name}: {roc_auc:.4f}")
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC Curve - Iris")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=150)
    plt.close()

    # 7. Let the user enter measurements for a new flower.
    print("\nTry predicting a new flower.")
    print("Enter measurements in this order:")
    print("sepal length, sepal width, petal length, petal width (cm)")
    try:
        values = []
        prompts = [
            "Sepal length (cm): ",
            "Sepal width (cm): ",
            "Petal length (cm): ",
            "Petal width (cm): "
        ]
        for prompt in prompts:
            values.append(float(input(prompt)))
        if any(v <= 0 for v in values):
            print("Measurements should be positive. Skipping custom prediction.")
        else:
            prediction = int(model.predict(pd.DataFrame([values], columns=X.columns))[0])
            probabilities = model.predict_proba(pd.DataFrame([values], columns=X.columns))[0]
            print("\nPredicted species:", target_names[prediction].title())
            print("Class probabilities:")
            for name, prob in zip(target_names, probabilities):
                print(f"  {name.title()}: {prob:.2%}")
    except (ValueError, EOFError):
        print("No valid custom measurements entered; evaluation is complete.")

    print(f"\nCharts saved in: {OUTPUT_DIR}/")
    print("Project completed successfully.")

if __name__ == "__main__":
    main()
