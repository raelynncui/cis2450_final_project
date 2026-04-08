"""
step2_build_movie_genre_dataset.py

Joins the MusicBrainz soundtrack intermediate CSVs (from step 1) with
AcousticBrainz audio features and TMDB movie metadata to build a dataset
for predicting movie/show genre from acoustic audio features.

Final output: CSV where each row is a soundtrack recording with:
  - Movie metadata (IMDB ID, title, movie genres from TMDB)
  - AcousticBrainz audio features (BPM, danceability, key, loudness, etc.)
  - One-hot encoded categorical features (key_key → 12 cols, key_scale → 1 col)
"""

import polars as pl
import glob
import os
import sys

# ============================================================================
# CONFIG
# ============================================================================

SOUNDTRACK_RGS_CSV = "data/mb_soundtrack_rgs.csv"
SOUNDTRACK_RECORDINGS_CSV = "data/mb_soundtrack_recordings.csv"
TMDB_CSV = "data/TMDB_movie_dataset_v11.csv" # not included because too large

# AcousticBrainz CSVs
AB_RHYTHM_CSV = "data/acousticbrainz-lowlevel-features-20220623-rhythm.csv" # not included because too large
AB_TONAL_CSV = "data/acousticbrainz-lowlevel-features-20220623-tonal.csv" # not included because too large
AB_LOWLEVEL_CSV = "data/acousticbrainz-lowlevel-features-20220623-lowlevel.csv" # not included because too large

OUTPUT_CSV = "data/movie_genre_audio_features_dataset.csv"


# ============================================================================
# Helpers
# ============================================================================

def find_ab_file(keyword: str) -> str:
    """Find an AcousticBrainz feature CSV by keyword at the END of the
    filename (before .csv), to avoid the bug where all three files
    match '*lowlevel*'."""
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


# ============================================================================
# Main pipeline
# ============================================================================

def main():
    print(f"{'='*60}")
    print("STEP 2: Building movie-genre + audio-features dataset")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Verify inputs
    # ------------------------------------------------------------------
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

    print(f"\n  Input files:")
    print(f"    Soundtrack RGs:        {SOUNDTRACK_RGS_CSV}")
    print(f"    Soundtrack recordings: {SOUNDTRACK_RECORDINGS_CSV}")
    print(f"    TMDB:                  {TMDB_CSV}")
    print(f"    AB rhythm:             {ab_rhythm or 'NOT FOUND'}")
    print(f"    AB tonal:              {ab_tonal or 'NOT FOUND'}")
    print(f"    AB lowlevel:           {ab_lowlevel or 'NOT FOUND'}")

    if not all([ab_rhythm, ab_tonal, ab_lowlevel]):
        print("\nERROR: Could not find all three AcousticBrainz CSV files.")
        sys.exit(1)

    # Sanity check: make sure all three paths are distinct files
    ab_paths = [os.path.abspath(p) for p in [ab_rhythm, ab_tonal, ab_lowlevel]]
    if len(set(ab_paths)) < 3:
        print("\nERROR: Two or more AcousticBrainz paths point to the same file!")
        print(f"  rhythm:   {ab_rhythm}")
        print(f"  tonal:    {ab_tonal}")
        print(f"  lowlevel: {ab_lowlevel}")
        print("Set the exact filenames in the CONFIG section.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load TMDB → movie genres keyed by imdb_id
    # ------------------------------------------------------------------
    print(f"\n--- Loading TMDB movie dataset ---")

    tmdb = pl.read_csv(TMDB_CSV, infer_schema_length=10000)
    print(f"  Raw TMDB rows: {tmdb.shape[0]:,}")

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

    if "tmdb_imdb_id" not in tmdb.columns:
        print(f"ERROR: No imdb_id column found. Available: {tmdb.columns}")
        sys.exit(1)

    keep_cols = [c for c in ["tmdb_imdb_id", "tmdb_id", "tmdb_title",
                             "movie_genres"] if c in tmdb.columns]
    tmdb = (
        tmdb.select(keep_cols)
        .with_columns(pl.col("tmdb_imdb_id").cast(pl.Utf8).str.strip_chars())
        .filter(pl.col("tmdb_imdb_id").str.starts_with("tt"))
        .unique(subset="tmdb_imdb_id", keep="first")
    )
    print(f"  TMDB movies with valid IMDB ID: {tmdb.shape[0]:,}")

    # ------------------------------------------------------------------
    # Load soundtrack release groups + join with TMDB
    # ------------------------------------------------------------------
    print(f"\n--- Loading soundtrack release groups ---")

    rgs = pl.read_csv(SOUNDTRACK_RGS_CSV)
    print(f"  Soundtrack RGs with IMDB links: {rgs.shape[0]:,}")

    rgs = rgs.join(tmdb, left_on="imdb_id", right_on="tmdb_imdb_id", how="inner")
    print(f"  After TMDB join: {rgs.shape[0]:,}")

    if rgs.shape[0] == 0:
        print("ERROR: No IMDB IDs matched between soundtrack RGs and TMDB.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load soundtrack recordings + join with release groups
    # ------------------------------------------------------------------
    print(f"\n--- Loading soundtrack recordings ---")

    recordings = pl.read_csv(SOUNDTRACK_RECORDINGS_CSV)
    print(f"  Raw recording rows: {recordings.shape[0]:,}")

    recordings = recordings.unique(subset=["recording_mbid", "rg_id"], keep="first")
    print(f"  After dedup: {recordings.shape[0]:,}")

    rg_cols = ["rg_id", "rg_title", "imdb_id", "movie_genres",
               *[c for c in ["tmdb_id", "tmdb_title"] if c in rgs.columns]]
    rec_with_movie = recordings.join(rgs.select(rg_cols), on="rg_id", how="inner")
    print(f"  Recordings matched to movies: {rec_with_movie.shape[0]:,}")

    rec_with_movie = rec_with_movie.unique(subset="recording_mbid", keep="first")
    print(f"  After dedup (one movie per recording): {rec_with_movie.shape[0]:,}")

    # ------------------------------------------------------------------
    # Load and merge AcousticBrainz feature CSVs
    # ------------------------------------------------------------------
    print(f"\n--- Loading AcousticBrainz feature CSVs ---")

    print(f"  Loading rhythm: {ab_rhythm}")
    ab_r = pl.read_csv(ab_rhythm)
    print(f"    Rows: {ab_r.shape[0]:,}  Columns: {ab_r.columns}")

    print(f"  Loading tonal: {ab_tonal}")
    ab_t = pl.read_csv(ab_tonal)
    print(f"    Rows: {ab_t.shape[0]:,}  Columns: {ab_t.columns}")

    print(f"  Loading lowlevel: {ab_lowlevel}")
    ab_l = pl.read_csv(ab_lowlevel)
    print(f"    Rows: {ab_l.shape[0]:,}  Columns: {ab_l.columns}")

    # Verify no unexpected column overlaps (beyond mbid and submission_offset)
    r_cols = set(ab_r.columns) - {"mbid", "submission_offset"}
    t_cols = set(ab_t.columns) - {"mbid", "submission_offset"}
    l_cols = set(ab_l.columns) - {"mbid", "submission_offset"}
    overlaps = (r_cols & t_cols) | (r_cols & l_cols) | (t_cols & l_cols)
    if overlaps:
        print(f"\n  WARNING: Unexpected column overlaps across AB files: {overlaps}")
        print("  These will produce duplicate-suffix columns in the join.")

    # Filter to primary submissions only
    ab_r = ab_r.filter(pl.col("submission_offset") == 0).drop("submission_offset")
    ab_t = ab_t.filter(pl.col("submission_offset") == 0).drop("submission_offset")
    ab_l = ab_l.filter(pl.col("submission_offset") == 0).drop("submission_offset")

    print(f"\n  After offset=0 filter:")
    print(f"    Rhythm:   {ab_r.shape[0]:,}  cols: {ab_r.columns}")
    print(f"    Tonal:    {ab_t.shape[0]:,}  cols: {ab_t.columns}")
    print(f"    Lowlevel: {ab_l.shape[0]:,}  cols: {ab_l.columns}")

    # Join all three on mbid
    # Using suffix to make accidental overlaps visible instead of silent
    ab_features = (
        ab_r
        .join(ab_t, on="mbid", how="inner", suffix="_TONAL_DUP")
        .join(ab_l, on="mbid", how="inner", suffix="_LOWLEVEL_DUP")
    )

    # Drop any accidental duplicate columns
    dup_cols = [c for c in ab_features.columns if c.endswith("_DUP")]
    if dup_cols:
        print(f"\n  WARNING: Dropping duplicate columns from join: {dup_cols}")
        ab_features = ab_features.drop(dup_cols)

    print(f"\n  Merged AcousticBrainz features: {ab_features.shape[0]:,} rows")
    print(f"  Columns: {ab_features.columns}")

    # ------------------------------------------------------------------
    # Final join: recordings ↔ AcousticBrainz
    # ------------------------------------------------------------------
    print(f"\n--- Final join: recordings ↔ AcousticBrainz ---")

    rec_with_movie = rec_with_movie.with_columns(
        pl.col("recording_mbid").str.to_lowercase()
    )
    ab_features = ab_features.with_columns(
        pl.col("mbid").str.to_lowercase()
    )

    final = rec_with_movie.join(
        ab_features, left_on="recording_mbid", right_on="mbid", how="inner"
    )

    print(f"  Matched rows:        {final.shape[0]:,}")
    print(f"  Unique recordings:   {final['recording_mbid'].n_unique():,}")
    print(f"  Unique movies/shows: {final['imdb_id'].n_unique():,}")

    if final.shape[0] == 0:
        print("\nERROR: No matches. Check MBID samples:")
        print(f"  Recordings: {rec_with_movie['recording_mbid'].head(3).to_list()}")
        print(f"  AcousticBrainz: {ab_features['mbid'].head(3).to_list()}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Process movie genres
    # ------------------------------------------------------------------
    print(f"\n--- Processing movie genres ---")

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

    print(f"  Genre distribution (top 20):")
    print(
        final.filter(pl.col("primary_movie_genre").is_not_null()
                     & (pl.col("primary_movie_genre") != ""))
        .group_by("primary_movie_genre").len()
        .sort("len", descending=True).head(20)
    )

    # ------------------------------------------------------------------
    # One-hot encode categorical features (key_key, key_scale)
    # ------------------------------------------------------------------
    print(f"\n--- One-hot encoding categorical features ---")

    if "key_key" in final.columns:
        unique_keys = sorted(final["key_key"].drop_nulls().unique().to_list())
        print(f"  key_key values ({len(unique_keys)}): {unique_keys}")
        for k in unique_keys:
            final = final.with_columns(
                (pl.col("key_key") == k).cast(pl.Int8).alias(f"key_{k}")
            )
        final = final.drop("key_key")
        print(f"  Created {len(unique_keys)} one-hot columns (key_A, key_B, ...)")

    if "key_scale" in final.columns:
        unique_scales = sorted(final["key_scale"].drop_nulls().unique().to_list())
        print(f"  key_scale values: {unique_scales}")
        final = final.with_columns(
            (pl.col("key_scale") == "minor").cast(pl.Int8).alias("scale_minor")
        )
        final = final.drop("key_scale")
        print(f"  Created 1 column: scale_minor (1=minor, 0=major)")

    # ------------------------------------------------------------------
    # Select and order columns, write output
    # ------------------------------------------------------------------
    print(f"\n--- Writing final dataset ---")

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

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_rows = final.shape[0]
    n_recordings = final["recording_mbid"].n_unique()
    n_movies = final["imdb_id"].n_unique()
    n_genres = (final["primary_movie_genre"].n_unique()
                if "primary_movie_genre" in final.columns else 0)

    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}")
    print(f"  Output:              {OUTPUT_CSV}")
    print(f"  Total rows:          {n_rows:,}")
    print(f"  Unique recordings:   {n_recordings:,}")
    print(f"  Unique movies/shows: {n_movies:,}")
    print(f"  Unique movie genres: {n_genres:,}")

    print(f"\n  All columns in output ({len(final_cols)}):")
    for c in final_cols:
        print(f"    {c:50s}  {final[c].dtype}")

    if n_rows >= 50_000:
        print(f"\n  ✓ Dataset exceeds 50k rows!")
    else:
        print(f"\n  ⚠ Only {n_rows:,} rows — below the 50k target.")

    print(f"\n  Sample rows:")
    sample_cols = [c for c in ["recording_mbid", "track_title", "artist",
                                "imdb_id", "primary_movie_genre", "bpm",
                                "danceability", "average_loudness", "scale_minor"]
                   if c in final.columns]
    print(final.select(sample_cols).head(10))


if __name__ == "__main__":
    main()