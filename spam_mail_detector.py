"""
QSkill Internship Project 2: Spam Mail Detector
Run: python spam_mail_detector.py

Dataset:
SMS Spam Collection (UCI mirror). The script attempts to download it automatically.
If download fails, download the SMS Spam Collection file and save it as
SMSSpamCollection.txt beside this script. The file is tab-separated: label<TAB>message.
"""

import os
import urllib.request
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

DATA_FILE = "SMSSpamCollection.txt"
DATA_URLS = [
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
    "https://raw.githubusercontent.com/selva86/datasets/master/sms_spam.csv"
]
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_dataset():
    """Load local dataset first; otherwise attempt known public mirrors."""
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE, sep="\t", header=None, names=["label", "message"], encoding="utf-8")
        df = df.dropna(subset=["label", "message"])
        return normalize_dataset(df)

    last_error = None
    for url in DATA_URLS:
        try:
            print("Trying to download dataset from:", url)
            local_path = os.path.join(OUTPUT_DIR, "downloaded_sms_data.csv")
            urllib.request.urlretrieve(url, local_path)
            try:
                df = pd.read_csv(local_path, sep="\t")
                if {"v1", "v2"}.issubset(df.columns):
                    df = df.rename(columns={"v1": "label", "v2": "message"})
                elif {"label", "message"}.issubset(df.columns):
                    pass
                elif df.shape[1] >= 2:
                    df = df.iloc[:, :2]
                    df.columns = ["label", "message"]
                return normalize_dataset(df)
            except Exception as exc:
                last_error = exc
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "Could not download the dataset. Please place the UCI SMS Spam Collection "
        "file named 'SMSSpamCollection.txt' beside this script. "
        f"Last error: {last_error}"
    )

def normalize_dataset(df):
    df = df[["label", "message"]].copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()
    df["label"] = df["label"].replace({"ham": "ham", "spam": "spam"})
    df = df[df["label"].isin(["ham", "spam"])]
    df = df.dropna().drop_duplicates()
    if df.empty:
        raise ValueError("Dataset is empty or its columns/labels are not in the expected format.")
    return df

def main():
    print("=" * 60)
    print("QSKILL INTERNSHIP: SPAM MAIL DETECTOR")
    print("=" * 60)

    # 1. Load and inspect text data.
    df = load_dataset()
    print("\nFirst five messages:")
    print(df.head().to_string(index=False))
    print("\nDataset shape:", df.shape)
    print("\nLabel counts:")
    print(df["label"].value_counts().to_string())

    # Save a label distribution chart.
    df["label"].value_counts().reindex(["ham", "spam"]).plot(
        kind="bar", title="Spam vs Not Spam Message Counts", rot=0
    )
    plt.xlabel("Message category")
    plt.ylabel("Number of messages")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "label_distribution.png"), dpi=150)
    plt.close()

    # 2. Split into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.20,
        random_state=42, stratify=df["label"]
    )

    # 3. Convert text to TF-IDF features and train Naive Bayes.
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, stop_words="english",
            ngram_range=(1, 2), max_features=30000
        )),
        ("classifier", MultinomialNB())
    ])
    model.fit(X_train, y_train)

    # 4. Evaluate.
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=["ham", "spam"], zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
    print("Confusion matrix (rows = actual, columns = predicted; ham, spam):")
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Spam", "Spam"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Spam Mail Detector - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # 5. Try user-entered messages.
    print("\nTest your own message. Type 'quit' to stop.")
    while True:
        try:
            message = input("\nEnter a message: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting message tester.")
            break
        if message.lower() in {"quit", "exit", "q"}:
            break
        if not message:
            print("Please enter a non-empty message.")
            continue

        prediction = model.predict([message])[0]
        probabilities = model.predict_proba([message])[0]
        classes = list(model.named_steps["classifier"].classes_)
        confidence = float(probabilities[classes.index(prediction)])
        result = "SPAM" if prediction == "spam" else "NOT SPAM (HAM)"
        print(f"Prediction: {result}")
        print(f"Model confidence: {confidence:.2%}")
        print("Note: This is a machine-learning prediction, not a guarantee.")

    print(f"\nChart saved in: {OUTPUT_DIR}/")
    print("Project completed successfully.")

if __name__ == "__main__":
    main()
