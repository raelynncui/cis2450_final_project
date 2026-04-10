"""
step1_parse_soundtracks.py

Parses MusicBrainz JSON dumps to extract:
  1A. Soundtrack release groups with IMDB links  -->  mb_soundtrack_rgs.csv
  1B. Recording MBIDs for those release groups   -->  mb_soundtrack_recordings.csv
"""

import json
import tarfile
import csv
import os
import re
import sys
from tqdm import tqdm


MB_RELEASE_GROUP_PATH = "release-group.tar.xz"
MB_RELEASE_PATH = "release.tar.xz"
OUT_SOUNDTRACK_RGS = "mb_soundtrack_rgs.csv"
OUT_SOUNDTRACK_RECORDINGS = "mb_soundtrack_recordings.csv"


def extract_imdb_id(url: str) -> str | None:
    m = re.search(r"(tt\d{7,})", url)
    return m.group(1) if m else None


# Extract artist string from an artist-credit array
def _extract_artist(artist_credit) -> str:
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


"""
    Parse release-group.tar.xz to find soundtrack release groups with IMDB links.
    Outputs a CSV with columns: rg_id, rg_title, imdb_id, primary_type
"""

def parse_release_groups(tar_path: str, out_csv: str):
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

                    # Check if this is a soundtrack
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

                    # Look for IMDB URL 
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



"""
Load mapping of release group ID --> imdb_id from previous step output
"""
def _load_soundtrack_rg_ids(rgs_csv: str) -> dict:
    rg_map = {}
    with open(rgs_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rg_map[row["rg_id"]] = row["imdb_id"]
    return rg_map


"""
Parse through release.tar.xz (only process releases whose release group is in the soundtrack set from previous step 1A)
Then, extract recording MBIDs and track metadata

Outputs a CSV columns with: recording_mbid, rg_id, imdb_id, track_title, artist, track_position
"""

def parse_releases(tar_path: str, out_csv: str, rgs_csv: str):

    # Load the soundtrack release group IDs from previous step
    rg_map = _load_soundtrack_rg_ids(rgs_csv)

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


def main():
    if os.path.exists(OUT_SOUNDTRACK_RGS):
        # File already exists — skip parsing release groups
        pass
    else:
        if not os.path.exists(MB_RELEASE_GROUP_PATH):
            print(f"ERROR: {MB_RELEASE_GROUP_PATH} not found.")
            sys.exit(1)
        parse_release_groups(MB_RELEASE_GROUP_PATH, OUT_SOUNDTRACK_RGS)

    if os.path.exists(OUT_SOUNDTRACK_RECORDINGS):
        # File already exists — skip parsing releases
        pass
    else:
        if not os.path.exists(MB_RELEASE_PATH):
            print(f"ERROR: {MB_RELEASE_PATH} not found.")
            sys.exit(1)
        if not os.path.exists(OUT_SOUNDTRACK_RGS):
            print(f"ERROR: {OUT_SOUNDTRACK_RGS} not found.")
            sys.exit(1)
        parse_releases(MB_RELEASE_PATH, OUT_SOUNDTRACK_RECORDINGS,
                       OUT_SOUNDTRACK_RGS)


if __name__ == "__main__":
    main()