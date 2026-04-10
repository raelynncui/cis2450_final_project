"""
step2_build_movie_genre_dataset.py

Joins the MusicBrainz soundtrack intermediate CSVs (from step 1) with AcousticBrainz audio features 
and TMDB movie metadata to build a dataset for predicting movie/show genre from acoustic audio features.

Final output: CSV where each row is a soundtrack recording with:
  - Movie metadata (IMDB ID, title, movie genres from TMDB)
  - AcousticBrainz audio features (BPM, danceability, key, loudness, etc.)
  - One-hot encoded categorical features
"""

import polars as pl
import glob
import os
import sys


SOUNDTRACK_RGS_CSV = "data/mb_soundtrack_rgs.csv"
SOUNDTRACK_RECORDINGS_CSV = "data/mb_soundtrack_recordings.csv"
TMDB_CSV = "data/TMDB_movie_dataset_v11.csv" # not included because too large

# AcousticBrainz CSVs
AB_RHYTHM_CSV = "data/acousticbrainz-lowlevel-features-20220623-rhythm.csv" # not included because too large
AB_TONAL_CSV = "data/acousticbrainz-lowlevel-features-20220623-tonal.csv" # not included because too large
AB_LOWLEVEL_CSV = "data/acousticbrainz-lowlevel-features-20220623-lowlevel.csv" # not included because too large

OUTPUT_CSV = "data/movie_genre_audio_features_dataset.csv"



"""
Finds all AcousticBrainz feature CSVs
"""
def find_ab_file(keyword: str) -> str:
    patterns = [
        f"*-{keyword}.csv",
        f"*_{keyword}.csv",
        f"*acousticbrainz*{keyword}*",
    ]
    exclude = {"genre_audio", "mb_", "movie", "spotify", "TMDB"}
    for pat in patterns:
        matches = glob.glob(pat)
        matches = [m for m in matches if not any(ex in m for ex in exclude)]
        if len(matches) == 1:
            return matches[0]
        if matches:
            exact = [m for m in matches if m.rsplit(".", 1)[0].endswith(keyword)]
            if exact:
                return exact[0]
            return matches[0]
    return ""


def main():
    # Verify inputs
    for f in [SOUNDTRACK_RGS_CSV, SOUNDTRACK_RECORDINGS_CSV]:
        if not os.path.exists(f):
            print(f"ERROR: {f} not found. Run step1_parse_soundtracks.py first.")
            sys.exit(1)

    if not os.path.exists(TMDB_CSV):
        print(f"ERROR: {TMDB_CSV} not found.")
        print("Download from: https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies")
        sys.exit(1)

    ab_rhythm = AB_RHYTHM_CSV if os.path.exists(AB_RHYTHM_CSV) else find_ab_file("rhythm")
    ab_tonal = AB_TONAL_CSV if os.path.exists(AB_TONAL_CSV) else find_ab_file("tonal")
    ab_lowlevel = AB_LOWLEVEL_CSV if os.path.exists(AB_LOWLEVEL_CSV) else find_ab_file("lowlevel")

    # Load TMDB --> movie genres keyed by imdb_id
    tmdb = pl.read_csv(TMDB_CSV, infer_schema_length=10000)

    col_renames = {}
    for c in tmdb.columns:
        cl = c.lower().strip()
        if cl in ("imdb_id", "imdb id"):
            col_renames[c] = "tmdb_imdb_id"
        elif cl in ("id", "tmdb_id"):
            col_renames[c] = "tmdb_id"
        elif cl in ("title", "original_title") and "tmdb_title" not in col_renames.values():
            col_renames[c] = "tmdb_title"
        elif cl in ("genres", "genre"):
            col_renames[c] = "movie_genres"

    tmdb = tmdb.rename(col_renames)

    keep_cols = [c for c in ["tmdb_imdb_id", "tmdb_id", "tmdb_title",
                             "movie_genres"] if c in tmdb.columns]
    tmdb = (
        tmdb.select(keep_cols)
        .with_columns(pl.col("tmdb_imdb_id").cast(pl.Utf8).str.strip_chars())
        .filter(pl.col("tmdb_imdb_id").str.starts_with("tt"))
        .unique(subset="tmdb_imdb_id", keep="first")
    )

    # Load soundtrack release groups + join with TMDB
    rgs = pl.read_csv(SOUNDTRACK_RGS_CSV)
    rgs = rgs.join(tmdb, left_on="imdb_id", right_on="tmdb_imdb_id", how="inner")

    # Load soundtrack recordings + join with release groups
    recordings = pl.read_csv(SOUNDTRACK_RECORDINGS_CSV)
    recordings = recordings.unique(subset=["recording_mbid", "rg_id"], keep="first")
    rg_cols = ["rg_id", "rg_title", "imdb_id", "movie_genres",
               *[c for c in ["tmdb_id", "tmdb_title"] if c in rgs.columns]]
    rec_with_movie = recordings.join(rgs.select(rg_cols), on="rg_id", how="inner")
    rec_with_movie = rec_with_movie.unique(subset="recording_mbid", keep="first")

    # Load and merge AcousticBrainz feature CSVs
    ab_r = pl.read_csv(ab_rhythm)
    ab_t = pl.read_csv(ab_tonal)
    ab_l = pl.read_csv(ab_lowlevel)

    # Join all three on mbid
    ab_features = (
        ab_r
        .join(ab_t, on="mbid", how="inner")
        .join(ab_l, on="mbid", how="inner")
    )

    # Final join: recordings with AcousticBrainz
    rec_with_movie = rec_with_movie.with_columns(
        pl.col("recording_mbid").str.to_lowercase()
    )
    ab_features = ab_features.with_columns(
        pl.col("mbid").str.to_lowercase()
    )
    final = rec_with_movie.join(
        ab_features, left_on="recording_mbid", right_on="mbid", how="inner"
    )

    # Process movie genres
    final = final.with_columns(
        pl.col("movie_genres")
          .cast(pl.Utf8)
          .str.replace_all(r"[\[\]'\"]", "")
          .str.strip_chars()
          .alias("movie_genres")
    )

    final = final.with_columns(
        pl.col("movie_genres")
          .str.split(",")
          .list.first()
          .str.strip_chars()
          .alias("primary_movie_genre")
    )


    # One-hot encode categorical features (key_key, key_scale)
    if "key_key" in final.columns:
        unique_keys = sorted(final["key_key"].drop_nulls().unique().to_list())
        for k in unique_keys:
            final = final.with_columns(
                (pl.col("key_key") == k).cast(pl.Int8).alias(f"key_{k}")
            )
        final = final.drop("key_key")

    if "key_scale" in final.columns:
        unique_scales = sorted(final["key_scale"].drop_nulls().unique().to_list())
        final = final.with_columns(
            (pl.col("key_scale") == "minor").cast(pl.Int8).alias("scale_minor")
        )
        final = final.drop("key_scale")

    # Select and order columns, write final dataset
    metadata_cols = [
        "recording_mbid", "track_title", "artist",
        "imdb_id", "rg_id", "rg_title",
        "primary_movie_genre", "movie_genres",
    ]
    for c in ["tmdb_id", "tmdb_title"]:
        if c in final.columns:
            metadata_cols.append(c)

    feature_cols = [c for c in final.columns
                    if c not in metadata_cols
                    and c not in ("track_position", "primary_type", "imdb_id_right")]

    desired_cols = metadata_cols + sorted(feature_cols)
    final_cols = [c for c in desired_cols if c in final.columns]
    final = final.select(final_cols)
    final.write_csv(OUTPUT_CSV)


if __name__ == "__main__":
    main()