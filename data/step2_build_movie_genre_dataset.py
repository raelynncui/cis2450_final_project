#!/usr/bin/env python3
"""
step2_build_movie_genre_dataset.py

Joins the MusicBrainz soundtrack intermediate CSVs (from step 1) with
AcousticBrainz audio features and TMDB movie metadata to build a dataset
for predicting movie/show genre from acoustic audio features.

Final output: CSV where each row is a soundtrack recording with:
  - Movie metadata (IMDB ID, title, movie genres from TMDB)
  - AcousticBrainz audio features (BPM, danceability, key, loudness, etc.)
"""

import polars as pl
import glob
import os
import sys

# ============================================================================
# CONFIG
# ============================================================================

# Intermediate CSVs from step 1
SOUNDTRACK_RGS_CSV = "mb_soundtrack_rgs.csv"
SOUNDTRACK_RECORDINGS_CSV = "mb_soundtrack_recordings.csv"

# TMDB movie dataset (Kaggle CSV)
TMDB_CSV = "TMDB_movie_dataset_v11.csv"

# AcousticBrainz feature CSVs (auto-detected by keyword)
AB_RHYTHM_CSV = ""    # leave blank for auto-detect
AB_TONAL_CSV = ""     # leave blank for auto-detect
AB_LOWLEVEL_CSV = ""  # leave blank for auto-detect

# Output
OUTPUT_CSV = "movie_genre_audio_features_dataset.csv"


# ============================================================================
# Helpers
# ============================================================================

def find_ab_file(keyword: str) -> str:
    """Find an AcousticBrainz feature CSV by keyword in the filename."""
    patterns = [
        f"*acousticbrainz*{keyword}*.csv",
        f"*{keyword}*.csv",
    ]
    for pat in patterns:
        matches = glob.glob(pat)
        matches = [m for m in matches if "genre_audio" not in m
                   and "mb_" not in m and "movie" not in m
                   and "spotify" not in m and "TMDB" not in m]
        if matches:
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

    ab_rhythm = AB_RHYTHM_CSV or find_ab_file("rhythm")
    ab_tonal = AB_TONAL_CSV or find_ab_file("tonal")
    ab_lowlevel = AB_LOWLEVEL_CSV or find_ab_file("lowlevel")

    print(f"\n  Input files:")
    print(f"    Soundtrack RGs:       {SOUNDTRACK_RGS_CSV}")
    print(f"    Soundtrack recordings:{SOUNDTRACK_RECORDINGS_CSV}")
    print(f"    TMDB:                 {TMDB_CSV}")
    print(f"    AB rhythm:            {ab_rhythm or 'NOT FOUND'}")
    print(f"    AB tonal:             {ab_tonal or 'NOT FOUND'}")
    print(f"    AB lowlevel:          {ab_lowlevel or 'NOT FOUND'}")

    if not all([ab_rhythm, ab_tonal, ab_lowlevel]):
        print("\nERROR: Could not find all three AcousticBrainz CSV files.")
        print("Expected files matching patterns like:")
        print("  *acousticbrainz*rhythm*.csv")
        print("  *acousticbrainz*tonal*.csv")
        print("  *acousticbrainz*lowlevel*.csv")
        print("\nPlace them in the current directory or set the paths in CONFIG.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load TMDB → get movie genres keyed by imdb_id
    # ------------------------------------------------------------------
    print(f"\n--- Loading TMDB movie dataset ---")

    tmdb = pl.read_csv(TMDB_CSV, infer_schema_length=10000)
    print(f"  Raw TMDB rows: {tmdb.shape[0]:,}")
    print(f"  Columns: {tmdb.columns}")

    # Standardize column names
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
        print("ERROR: No imdb_id column found in TMDB dataset!")
        print(f"  Available: {tmdb.columns}")
        sys.exit(1)

    # Keep only rows with valid IMDB IDs and relevant columns
    keep_cols = [c for c in ["tmdb_imdb_id", "tmdb_id", "tmdb_title",
                             "movie_genres"] if c in tmdb.columns]
    tmdb = (
        tmdb
        .select(keep_cols)
        .with_columns(pl.col("tmdb_imdb_id").cast(pl.Utf8).str.strip_chars())
        .filter(pl.col("tmdb_imdb_id").str.starts_with("tt"))
        .unique(subset="tmdb_imdb_id", keep="first")
    )
    print(f"  TMDB movies with valid IMDB ID: {tmdb.shape[0]:,}")

    # ------------------------------------------------------------------
    # Load soundtrack release groups from step 1A
    # ------------------------------------------------------------------
    print(f"\n--- Loading soundtrack release groups ---")

    rgs = pl.read_csv(SOUNDTRACK_RGS_CSV)
    print(f"  Soundtrack release groups with IMDB links: {rgs.shape[0]:,}")
    print(f"  Unique IMDB IDs: {rgs['imdb_id'].n_unique():,}")

    # Join with TMDB to get movie genres
    rgs = rgs.join(
        tmdb,
        left_on="imdb_id",
        right_on="tmdb_imdb_id",
        how="inner",
    )
    print(f"  After TMDB join (have movie genres): {rgs.shape[0]:,}")

    if rgs.shape[0] == 0:
        print("\nERROR: No IMDB IDs matched between soundtrack RGs and TMDB.")
        print("  Check that IMDB ID formats match (both should be 'tt1234567').")
        print(f"  Sample from soundtracks: {pl.read_csv(SOUNDTRACK_RGS_CSV)['imdb_id'].head(5).to_list()}")
        print(f"  Sample from TMDB: {tmdb['tmdb_imdb_id'].head(5).to_list()}")
        sys.exit(1)

    # Show genre distribution of movies we matched
    print(f"\n  Movie genre distribution (top 15, raw TMDB genre strings):")
    if "movie_genres" in rgs.columns:
        genre_counts = (
            rgs
            .with_columns(pl.col("movie_genres").cast(pl.Utf8).fill_null(""))
            .filter(pl.col("movie_genres") != "")
            .group_by("movie_genres")
            .len()
            .sort("len", descending=True)
            .head(15)
        )
        print(genre_counts)

    # ------------------------------------------------------------------
    # Load soundtrack recordings from step 1B
    # ------------------------------------------------------------------
    print(f"\n--- Loading soundtrack recordings ---")

    recordings = pl.read_csv(SOUNDTRACK_RECORDINGS_CSV)
    print(f"  Total recording rows (with possible dups): {recordings.shape[0]:,}")

    # Deduplicate: same recording appearing in multiple editions
    recordings = recordings.unique(
        subset=["recording_mbid", "rg_id"], keep="first"
    )
    print(f"  After dedup (unique recording-rg pairs):   {recordings.shape[0]:,}")
    print(f"  Unique recording MBIDs: {recordings['recording_mbid'].n_unique():,}")

    # Join recordings with release groups (which now have movie genres)
    rec_with_movie = recordings.join(
        rgs.select(["rg_id", "rg_title", "imdb_id", "movie_genres",
                     *[c for c in ["tmdb_id", "tmdb_title"] if c in rgs.columns]]),
        on="rg_id",
        how="inner",
    )
    print(f"  Recordings matched to movies with genres: {rec_with_movie.shape[0]:,}")

    # A recording can appear in multiple soundtrack albums (e.g. re-releases).
    # Keep one row per recording, preferring the first match.
    rec_with_movie = rec_with_movie.unique(
        subset="recording_mbid", keep="first"
    )
    print(f"  After dedup (one movie per recording): {rec_with_movie.shape[0]:,}")

    # ------------------------------------------------------------------
    # Load AcousticBrainz feature CSVs
    # ------------------------------------------------------------------
    print(f"\n--- Loading AcousticBrainz feature CSVs ---")
    print(f"  (Each file is ~29M rows, this may take a minute)")

    print(f"  Loading rhythm features: {ab_rhythm}")
    ab_r = pl.read_csv(ab_rhythm)
    print(f"    Rows: {ab_r.shape[0]:,}, Columns: {ab_r.columns}")

    print(f"  Loading tonal features: {ab_tonal}")
    ab_t = pl.read_csv(ab_tonal)
    print(f"    Rows: {ab_t.shape[0]:,}, Columns: {ab_t.columns}")

    print(f"  Loading lowlevel features: {ab_lowlevel}")
    ab_l = pl.read_csv(ab_lowlevel)
    print(f"    Rows: {ab_l.shape[0]:,}, Columns: {ab_l.columns}")

    # Keep only primary submissions (offset == 0) to avoid duplicates
    ab_r = ab_r.filter(pl.col("submission_offset") == 0)
    ab_t = ab_t.filter(pl.col("submission_offset") == 0)
    ab_l = ab_l.filter(pl.col("submission_offset") == 0)

    print(f"\n  After keeping only primary submissions (offset=0):")
    print(f"    Rhythm:   {ab_r.shape[0]:,}")
    print(f"    Tonal:    {ab_t.shape[0]:,}")
    print(f"    Lowlevel: {ab_l.shape[0]:,}")

    # Merge the three feature files on mbid
    ab_r = ab_r.drop("submission_offset")
    ab_t = ab_t.drop("submission_offset").rename({"mbid": "mbid_t"})
    ab_l = ab_l.drop("submission_offset").rename({"mbid": "mbid_l"})

    ab_features = ab_r.join(
        ab_t, left_on="mbid", right_on="mbid_t", how="inner"
    ).join(
        ab_l, left_on="mbid", right_on="mbid_l", how="inner"
    )

    print(f"\n  Merged AcousticBrainz features: {ab_features.shape[0]:,} rows")
    print(f"  Feature columns: {ab_features.columns}")

    # ------------------------------------------------------------------
    # Final join: soundtrack recordings ↔ AcousticBrainz features
    # ------------------------------------------------------------------
    print(f"\n--- Final join: soundtrack recordings ↔ AcousticBrainz features ---")

    # Normalize MBIDs to lowercase for matching
    rec_with_movie = rec_with_movie.with_columns(
        pl.col("recording_mbid").str.to_lowercase()
    )
    ab_features = ab_features.with_columns(
        pl.col("mbid").str.to_lowercase()
    )

    final = rec_with_movie.join(
        ab_features,
        left_on="recording_mbid",
        right_on="mbid",
        how="inner",
    )

    print(f"  Matched rows: {final.shape[0]:,}")
    print(f"  Unique recordings: {final['recording_mbid'].n_unique():,}")
    print(f"  Unique movies/shows: {final['imdb_id'].n_unique():,}")

    if final.shape[0] == 0:
        print("\nERROR: No matches found between soundtrack recordings and AcousticBrainz.")
        print("  This means none of the soundtrack recording MBIDs appear in AcousticBrainz.")
        print(f"  Recordings sample: {rec_with_movie['recording_mbid'].head(5).to_list()}")
        print(f"  AcousticBrainz sample: {ab_features.head(5).select('mbid').to_series().to_list()}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Explode TMDB multi-genre strings into a primary_movie_genre column
    # ------------------------------------------------------------------
    print(f"\n--- Processing movie genres ---")

    # TMDB genres are typically comma-separated or JSON-like strings.
    # We'll extract the first genre as primary and keep the full string.
    if "movie_genres" in final.columns:
        final = final.with_columns([
            # Clean up genre string (remove brackets/quotes if JSON-like)
            pl.col("movie_genres")
              .cast(pl.Utf8)
              .str.replace_all(r"[\[\]'\"]", "")
              .str.strip_chars()
              .alias("movie_genres"),
        ])

        # Primary genre = first in the comma-separated list
        final = final.with_columns(
            pl.col("movie_genres")
              .str.split(",")
              .list.first()
              .str.strip_chars()
              .alias("primary_movie_genre")
        )

        print(f"  Primary movie genre distribution (top 20):")
        genre_dist = (
            final
            .filter(pl.col("primary_movie_genre").is_not_null()
                    & (pl.col("primary_movie_genre") != ""))
            .group_by("primary_movie_genre")
            .len()
            .sort("len", descending=True)
            .head(20)
        )
        print(genre_dist)

    # ------------------------------------------------------------------
    # Clean up and write output
    # ------------------------------------------------------------------
    print(f"\n--- Writing final dataset ---")

    metadata_cols = [
        "recording_mbid", "track_title", "artist",
        "imdb_id", "rg_id", "rg_title",
        "primary_movie_genre", "movie_genres",
    ]
    # Add tmdb columns if present
    for c in ["tmdb_id", "tmdb_title"]:
        if c in final.columns:
            metadata_cols.append(c)

    feature_cols = [c for c in final.columns
                    if c not in metadata_cols
                    and c not in ("track_position", "primary_type",
                                  "imdb_id_right")]

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

    print(f"\n  Feature columns available:")
    for c in sorted(feature_cols):
        if c in final.columns:
            print(f"    {c:45s}  {final[c].dtype}")

    if n_rows >= 50_000:
        print(f"\n  ✓ Dataset exceeds 50k rows!")
    else:
        print(f"\n  ⚠ Only {n_rows:,} rows — below the 50k target.")
        print(f"    Possible fixes:")
        print(f"    1. Keep all submission_offsets (not just offset=0)")
        print(f"    2. Allow multiple genre rows per recording")
        print(f"       (one row per movie genre in the TMDB list)")
        print(f"    3. Include release groups without IMDB links")
        print(f"       and try fuzzy-matching titles to TMDB")

    print(f"\n  Sample rows:")
    sample_cols = [c for c in ["recording_mbid", "track_title", "artist",
                                "imdb_id", "primary_movie_genre", "bpm",
                                "danceability", "key_key", "average_loudness"]
                   if c in final.columns]
    print(final.select(sample_cols).head(10))


if __name__ == "__main__":
    main()