"""
Train a decision tree classifier on the movie genre dataset.

Usage:
    ./.venv/bin/python data/run_decision_tree.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

cache_dir = Path("data/.cache")
mpl_config_dir = Path("data/.mplconfig")
cache_dir.mkdir(parents=True, exist_ok=True)
mpl_config_dir.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir.resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir.resolve()))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


DATASET_CSV = "data/movie_genre_audio_features_dataset.csv"
DEFAULT_OUTPUT_DIR = "data/plots"
TARGET_COL = "primary_movie_genre"
NON_FEATURE_COLS = {
    "recording_mbid",
    "track_title",
    "artist",
    "imdb_id",
    "rg_id",
    "rg_title",
    "movie_genres",
    "tmdb_id",
    "tmdb_title",
    TARGET_COL,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a quick decision tree model.")
    parser.add_argument("--dataset", default=DATASET_CSV, help="Path to dataset CSV.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where result plots will be written.",
    )
    parser.add_argument("--max-depth", type=int, default=6, help="Tree max depth.")
    parser.add_argument(
        "--min-genre-count",
        type=int,
        default=50,
        help="Only keep genres with at least this many rows.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows reserved for test split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def save_feature_importance_plot(importances: pl.DataFrame, output_dir: Path) -> None:
    top = importances.head(10).sort("importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["feature"].to_list(), top["importance"].to_list(), color="steelblue")
    ax.set_title("Decision Tree Top Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(output_dir / "decision_tree_feature_importance.png", dpi=200)
    plt.close(fig)


def save_confusion_matrix_plot(
    y_test: list[str],
    preds: np.ndarray,
    class_labels: list[str],
    output_dir: Path,
) -> None:
    cm = confusion_matrix(y_test, preds, labels=class_labels, normalize="true")
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title("Decision Tree Normalized Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(class_labels)))
    ax.set_yticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=90, fontsize=8)
    ax.set_yticklabels(class_labels, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall")
    fig.tight_layout()
    fig.savefig(output_dir / "decision_tree_confusion_matrix.png", dpi=200)
    plt.close(fig)


def save_class_metrics_plot(report_df: pl.DataFrame, output_dir: Path) -> None:
    top = report_df.sort("support", descending=True).head(12).sort("f1-score")
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top["genre"].to_list(), top["f1-score"].to_list(), color="darkorange")
    ax.set_title("Per-Genre F1 Score (Top 12 by Support)")
    ax.set_xlabel("F1 score")
    ax.set_ylabel("Genre")
    ax.set_xlim(0, 1)

    for bar, support in zip(bars, top["support"].to_list()):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"n={int(support)}",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_dir / "decision_tree_class_f1.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = pl.read_csv(args.dataset)
    print(f"Rows: {df.height:,}")
    print(f"Columns: {df.width}")

    genre_counts = (
        df.filter(pl.col(TARGET_COL).is_not_null() & (pl.col(TARGET_COL) != ""))
        .group_by(TARGET_COL)
        .len()
        .rename({"len": "count"})
    )
    valid_genres = genre_counts.filter(pl.col("count") >= args.min_genre_count).select(TARGET_COL)
    df = df.join(valid_genres, on=TARGET_COL, how="inner")
    print(f"Rows after filtering rare genres (< {args.min_genre_count}): {df.height:,}")

    feature_cols = [
        col
        for col, dtype in df.schema.items()
        if col not in NON_FEATURE_COLS and dtype.is_numeric()
    ]
    if not feature_cols:
        raise ValueError("No numeric feature columns were found.")

    print(f"Using {len(feature_cols)} numeric features.")

    model_df = df.select(feature_cols + [TARGET_COL]).with_columns(
        [
            pl.col(col).fill_null(pl.col(col).median()).fill_nan(pl.col(col).median())
            for col in feature_cols
        ]
    )
    X = model_df.select(feature_cols).to_numpy()
    y = model_df[TARGET_COL].to_list()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    clf = DecisionTreeClassifier(
        max_depth=args.max_depth,
        min_samples_leaf=5,
        random_state=args.random_state,
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    class_labels = sorted(set(y))
    report = classification_report(y_test, preds, zero_division=0, output_dict=True)

    print("\nQuick decision tree results")
    print("=" * 40)
    print(f"Train rows: {len(X_train):,}")
    print(f"Test rows:  {len(X_test):,}")
    print(f"Classes:    {len(set(y))}")
    print(f"Accuracy:   {accuracy:.4f}")

    importances = (
        pl.DataFrame(
            {
                "feature": feature_cols,
                "importance": clf.feature_importances_,
            }
        )
        .sort("importance", descending=True)
        .filter(pl.col("importance") > 0)
    )

    print("\nTop feature importances:")
    if importances.height == 0:
        print("  No non-zero feature importances.")
    else:
        print(importances.head(10))

    report_rows = []
    for label in class_labels:
        metrics = report[label]
        report_rows.append(
            {
                "genre": label,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1-score": metrics["f1-score"],
                "support": metrics["support"],
            }
        )
    report_df = pl.DataFrame(report_rows)

    save_feature_importance_plot(importances, output_dir)
    save_confusion_matrix_plot(y_test, preds, class_labels, output_dir)
    save_class_metrics_plot(report_df, output_dir)

    print(f"\nSaved plots to: {output_dir}")
    print("  - decision_tree_feature_importance.png")
    print("  - decision_tree_confusion_matrix.png")
    print("  - decision_tree_class_f1.png")

    print("\nClassification report:")
    print(classification_report(y_test, preds, zero_division=0))


if __name__ == "__main__":
    main()
