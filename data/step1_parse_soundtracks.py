"""
step1_parse_soundtracks.py

Parses MusicBrainz JSON dumps to extract:
  1A. Soundtrack release groups with IMDB links  →  mb_soundtrack_rgs.csv
  1B. Recording MBIDs for those release groups   →  mb_soundtrack_recordings.csv
"""

import json
import tarfile
import csv
import os
import re
import sys
from tqdm import tqdm

# ============================================================================
# CONFIG
# ============================================================================
MB_RELEASE_GROUP_PATH = "release-group.tar.xz"
MB_RELEASE_PATH = "release.tar.xz"

OUT_SOUNDTRACK_RGS = "mb_soundtrack_rgs.csv"
OUT_SOUNDTRACK_RECORDINGS = "mb_soundtrack_recordings.csv"


# ============================================================================
# Helpers
# ============================================================================

def extract_imdb_id(url: str) -> str | None:
    """Extract an IMDB title ID (e.g. 'tt0111161') from an IMDB URL."""
    m = re.search(r"(tt\d{7,})", url)
    return m.group(1) if m else None


def _extract_artist(artist_credit) -> str:
    """Extract a human-readable artist string from an artist-credit array."""
    if not artist_credit or not isinstance(artist_credit, list):
        return ""
    parts = []
    for ac in artist_credit:
        if isinstance(ac, str):
            parts.append(ac)
        elif isinstance(ac, dict):
            name = ac.get("name", "")
            if not name:
                artist_obj = ac.get("artist", {})
                name = artist_obj.get("name", "") if isinstance(artist_obj, dict) else ""
            parts.append(name)
            joinphrase = ac.get("joinphrase", "")
            if joinphrase:
                parts.append(joinphrase)
    return "".join(parts).strip()


# ============================================================================
# STEP 1A: Parse release-group dump → soundtrack RGs with IMDB links
# ============================================================================

def parse_release_groups(tar_path: str, out_csv: str):
    """
    Stream through release-group.tar.xz.
    Find release groups that:
      - Have secondary-type "Soundtrack"
      - Have an IMDB URL in their relations

    Output CSV columns:
        rg_id, rg_title, imdb_id, primary_type
    """
    print(f"\n{'='*60}")
    print("STEP 1A: Parsing release-group dump for soundtracks with IMDB links")
    print(f"{'='*60}")
    print(f"  File: {tar_path}")

    total = 0
    soundtrack_count = 0
    imdb_match_count = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["rg_id", "rg_title", "imdb_id", "primary_type"])

        with tarfile.open(tar_path, "r:xz") as tar:
            for member in tar:
                if member.name != "mbdump/release-group":
                    continue

                f = tar.extractfile(member)
                if f is None:
                    continue

                for line in tqdm(f, desc="  Release groups", unit=" lines"):
                    total += 1
                    try:
                        rg = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # ---- Check if this is a soundtrack ----
                    secondary_types = rg.get("secondary-types", [])
                    if not secondary_types:
                        stl = rg.get("secondary-type-list", {})
                        if isinstance(stl, dict):
                            secondary_types = stl.get("secondary-type", [])
                            if isinstance(secondary_types, str):
                                secondary_types = [secondary_types]

                    is_soundtrack = any(
                        t.lower() == "soundtrack" for t in secondary_types
                    )
                    if not is_soundtrack:
                        continue
                    soundtrack_count += 1

                    # ---- Look for IMDB URL in relations ----
                    relations = rg.get("relations", [])
                    imdb_id = None
                    for rel in relations:
                        if rel.get("target-type") == "url" or rel.get("type") == "IMDb":
                            url_obj = rel.get("url", {})
                            url_str = ""
                            if isinstance(url_obj, dict):
                                url_str = url_obj.get("resource", "")
                            elif isinstance(url_obj, str):
                                url_str = url_obj
                            if not url_str:
                                url_str = str(rel.get("target", ""))

                            if "imdb.com" in url_str:
                                imdb_id = extract_imdb_id(url_str)
                                if imdb_id:
                                    break

                    if imdb_id:
                        imdb_match_count += 1
                        rg_id = rg.get("id", "")
                        rg_title = rg.get("title", "")
                        primary_type = rg.get("primary-type", "")
                        writer.writerow([rg_id, rg_title, imdb_id, primary_type])

    print(f"\n  Results:")
    print(f"    Total release groups scanned:        {total:,}")
    print(f"    Soundtrack release groups found:      {soundtrack_count:,}")
    print(f"    Soundtracks with IMDB link (kept):   {imdb_match_count:,}")
    print(f"    Output: {out_csv}")


# ============================================================================
# STEP 1B: Parse release dump → recording MBIDs for soundtrack RGs
# ============================================================================

def _load_soundtrack_rg_ids(rgs_csv: str) -> dict:
    """Load mapping of release group ID → imdb_id from step 1A output."""
    rg_map = {}
    with open(rgs_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rg_map[row["rg_id"]] = row["imdb_id"]
    return rg_map


def parse_releases(tar_path: str, out_csv: str, rgs_csv: str):
    """
    Stream through release.tar.xz.
    Only process releases whose release group is in the soundtrack set
    from step 1A. Extract recording MBIDs and track metadata.

    Output CSV columns:
        recording_mbid, rg_id, imdb_id, track_title, artist, track_position
    """
    print(f"\n{'='*60}")
    print("STEP 1B: Parsing release dump for soundtrack recording MBIDs")
    print(f"{'='*60}")
    print(f"  File: {tar_path}")
    print("  (This file is ~335 GB uncompressed — expect 1-3 hours)")

    # Load the soundtrack release group IDs from step 1A
    print(f"  Loading soundtrack release group IDs from {rgs_csv}...")
    rg_map = _load_soundtrack_rg_ids(rgs_csv)
    print(f"  Soundtrack release groups to match: {len(rg_map):,}")

    total = 0
    matched_releases = 0
    track_rows = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["recording_mbid", "rg_id", "imdb_id",
                         "track_title", "artist", "track_position"])

        with tarfile.open(tar_path, "r:xz") as tar:
            for member in tar:
                if member.name != "mbdump/release":
                    continue

                f = tar.extractfile(member)
                if f is None:
                    continue

                for line in tqdm(f, desc="  Releases", unit=" lines"):
                    total += 1
                    try:
                        rel = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    rg = rel.get("release-group", {})
                    rg_id = rg.get("id", "")

                    # Skip releases not in our soundtrack set
                    if rg_id not in rg_map:
                        continue

                    matched_releases += 1
                    imdb_id = rg_map[rg_id]

                    media_list = rel.get("media", [])
                    for medium in media_list:
                        tracks = medium.get("tracks", [])
                        if not tracks:
                            tracks = medium.get("track-list", [])

                        for track in tracks:
                            recording = track.get("recording", {})
                            rec_mbid = recording.get("id", "")
                            if not rec_mbid:
                                continue

                            track_title = (track.get("title", "")
                                           or recording.get("title", ""))
                            track_pos = track.get("position",
                                                  track.get("number", ""))

                            artist = _extract_artist(
                                track.get("artist-credit", [])
                            )
                            if not artist:
                                artist = _extract_artist(
                                    recording.get("artist-credit", [])
                                )
                            if not artist:
                                artist = _extract_artist(
                                    rel.get("artist-credit", [])
                                )

                            writer.writerow([rec_mbid, rg_id, imdb_id,
                                             track_title, artist, track_pos])
                            track_rows += 1

                    if total % 1_000_000 == 0:
                        print(f"    ...processed {total:,} releases, "
                              f"{matched_releases:,} matched, "
                              f"{track_rows:,} track rows so far")

    print(f"\n  Results:")
    print(f"    Total releases scanned:           {total:,}")
    print(f"    Matched soundtrack releases:       {matched_releases:,}")
    print(f"    Recording rows written:            {track_rows:,}")
    print(f"    Output: {out_csv}")
    print(f"\n  Note: duplicates will be removed in step 2 using Polars.")


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Step 1A
    if os.path.exists(OUT_SOUNDTRACK_RGS):
        print(f"  {OUT_SOUNDTRACK_RGS} already exists — skipping step 1A.")
        print(f"  (Delete it to re-run)")
    else:
        if not os.path.exists(MB_RELEASE_GROUP_PATH):
            print(f"ERROR: {MB_RELEASE_GROUP_PATH} not found.")
            print("Update MB_RELEASE_GROUP_PATH in the CONFIG section.")
            sys.exit(1)
        parse_release_groups(MB_RELEASE_GROUP_PATH, OUT_SOUNDTRACK_RGS)

    # Step 1B
    if os.path.exists(OUT_SOUNDTRACK_RECORDINGS):
        print(f"\n  {OUT_SOUNDTRACK_RECORDINGS} already exists — skipping step 1B.")
        print(f"  (Delete it to re-run)")
    else:
        if not os.path.exists(MB_RELEASE_PATH):
            print(f"ERROR: {MB_RELEASE_PATH} not found.")
            print("Update MB_RELEASE_PATH in the CONFIG section.")
            sys.exit(1)
        if not os.path.exists(OUT_SOUNDTRACK_RGS):
            print(f"ERROR: {OUT_SOUNDTRACK_RGS} not found. Run step 1A first.")
            sys.exit(1)
        parse_releases(MB_RELEASE_PATH, OUT_SOUNDTRACK_RECORDINGS,
                       OUT_SOUNDTRACK_RGS)

    print(f"\n{'='*60}")
    print("STEP 1 COMPLETE")
    print(f"{'='*60}")
    print(f"  Intermediate files ready for step 2:")
    print(f"    {OUT_SOUNDTRACK_RGS}")
    print(f"    {OUT_SOUNDTRACK_RECORDINGS}")
    print(f"\n  Next: python step2_build_movie_genre_dataset.py")


if __name__ == "__main__":
    main()