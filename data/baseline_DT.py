import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def make_dataset(dataset_path: str):
    df = pl.read_csv(dataset_path)
    target_col = "primary_movie_genre"
    feature_cols = [
        col for col in df.columns if col not in [target_col] and df[col].dtype.is_numeric()
    ]
    df = df.select(feature_cols + [target_col])
    X = df.select(feature_cols).to_numpy()
    y = df[target_col].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_decision_tree(X_train: np.ndarray, y_train: np.ndarray, max_depth: int = 5, random_state: int = 42):
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(model, X_test: np.ndarray, y_test: np.ndarray):
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    report = classification_report(y_test, preds, zero_division=0, output_dict=True)
    confusion = confusion_matrix(y_test, preds, labels=model.classes_, normalize="true")
    return {"accuracy": accuracy, "f1": f1, "report": report, "confusion_matrix": confusion}

def main():
    path = "data/movie_genre_audio_features_dataset.csv"
    X_train, X_test, y_train, y_test = make_dataset(path)
    model = train_decision_tree(X_train, y_train)
    results = evaluate_classifier(model, X_test, y_test)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Weighted F1: {results['f1']:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, model.predict(X_test), zero_division=0))
    print("Confusion matrix:")
    print(results['confusion_matrix'])

if __name__ == "__main__":
    main()
