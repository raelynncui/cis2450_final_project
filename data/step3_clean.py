"""
step3_clean.py

This file cleans the movie_genre_audio_features_dataset.csv based on our initial EDA:
    - Removes null rows

Outputs movie_genre_audio_features_cleaned.csv
"""

import pandas as pd

# Uncleaned dataset
df = pd.read_csv("data/movie_genre_audio_features_dataset.csv")


# Remove nulls
df = df.dropna()
df.to_csv("data/movie_genre_audio_features_cleaned.csv", index=False)