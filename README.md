
# Predicting Movie Genre from Soundtrack Audio Features

Teresa Shang, Raelynn Cui

CIS 2450 Spring 2026

## Overview

This project investigates whether the acoustic properties of a movie's soundtrack can predict the genre of the movie itself. Rather than analyzing plot summaries, cast metadata, or poster imagery, we ask: does an action movie *sound* different from a drama at the level of raw audio features like tempo, danceability, and musical key?

We build a dataset by joining four public data sources on exact identifiers (IMDB IDs and MusicBrainz Recording IDs), resulting in a table where each row is a single soundtrack recording annotated with the parent movie's TMDB genre label and 25 acoustic features extracted by the AcousticBrainz project. We then train classification models to predict movie genre from these features alone.

## Data Sources
- MusicBrainz: https://data.metabrainz.org/pub/musicbrainz/data/json-dumps/
- AcousticBrainz: https://data.metabrainz.org/pub/musicbrainz/acousticbrainz/dumps/acousticbrainz-lowlevel-features-20220623/
- TMDB Movies Kaggle Dataset: https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies

## Pipeline

The data processing pipeline consists of three scripts:

**`step1_parse_soundtracks.py`** parses the two MusicBrainz tar archives. First, it parses through the release-group dump and filters for release groups with a secondary type of "Soundtrack" that also have an IMDB URL in their relations, outputting `mb_soundtrack_rgs.csv`. Then, it parses through the release dump and extracts recording MBIDs only for releases belonging to those soundtrack release groups, outputting `mb_soundtrack_recordings.csv`.

**`step2_build_movie_genre_dataset.py`** performs all joins using Polars. It loads the TMDB Kaggle CSV and joins it with the soundtrack release groups on IMDB ID to get movie genre labels. It then links recordings to their parent release groups, loads and merges the three AcousticBrainz feature CSVs on recording MBID, one-hot encodes the categorical features, and writes the final dataset.

**`step3_clean.py`** cleans the dataset to remove null rows. 

## LLM Usage

Claude was used across two chat sessions to build the data pipeline. The first session explored the MusicBrainz data format and produced early parsing scripts. The second session took those scripts, adapted them for genre prediction, and iterated through debugging.

All LLM-generated code was reviewed, tested, and modified before use.


#### Understanding the MusicBrainz file structure
 
> I want to build a dataset by joining a TMDB movie dump with MusicBrainz JSON dumps on IMDB ID. Analyze the feasibility. [Followed by debugging exchange where parsing output was pasted back, showing TIMESTAMP and COPYING files being parsed instead of the data file.]
 
Claude determined that the MusicBrainz tar archives contain metadata files (TIMESTAMP, COPYING, README) before the actual data file, and that the JSON follows MusicBrainz's web API format with field names like `secondary-types`, `relations[].url.resource`, and `media[].tracks[].recording.id`. Claude wrote diagnostic scripts to inspect the archive contents, which established that parsers must match on the exact path `member.name == "mbdump/release-group"` or `"mbdump/release"` rather than using loose string matching.

From this first session, `build_movie_songs_dataset.py`, `step1_parse_musicbrainz.py`, and `step2_build_dataset.py` scripts were generated.
 
#### Generating the extraction script
 
> Read through the project handoff summary to get context of what I'm working on. The file `build_movie_songs_dataset.py` has some logic to join soundtrack songs with movies/shows based on an IMDB ID, and the `step1_parse_musicbrainz.py` and `step2_build_dataset.py` files show the workflow described in the writeup.
 
The files provided were themselves outputs from the first chat session. Claude adapted the existing MusicBrainz parsing logic into `step1_parse_soundtracks.py`, which streams through the release-group archive to find soundtrack release groups with IMDB URL relations, then parses through the release archive to extract recording MBIDs only for those matched release groups. The two-step checkpoint design (outputting intermediate CSVs) was carried over from the earlier session, where it had been designed to solve crashes caused by in-memory deduplication of track pairs. For `step2_build_movie_genre_dataset.py`, Claude wrote the script for deduplicating recordings appearing in multiple release editions, extracting the primary movie genre from TMDB's comma-separated genre string, and writing the final dataset.

#### Validation
Each script was validated by inspecting intermediate outputs at every stage -- we checked that mb_soundtrack_rgs.csv contained valid IMDB IDs and reasonable row counts and checked that mb_soundtrack_recordings.csv only contained recordings belonging to release groups from the previous step. After joining with the AcousticBrainz CSVs, we confirmed all 12 expected audio features were present. In the EDA notebook, we received summary statistics, null counts, and distribution plots to confirm data values fell within plausible ranges.