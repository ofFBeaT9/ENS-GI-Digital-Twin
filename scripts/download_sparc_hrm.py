"""Download the SPARC Colonic HRM dataset.

Dataset: Dinning, Brookes et al. (2019/2023) — High resolution manometry,
         34 human ex vivo colon segments.
DOI:     https://doi.org/10.26275/RYFT-516S
License: CC BY

IMPORTANT: This dataset is EX VIVO (excised colon tissue in organ bath)
from colorectal cancer / diverticulitis patients — NOT IBS patients.
It is useful for testing the HRM loading pipeline and predict_manometry(),
but cannot be used for IBS classification validation.

Format: tab-delimited .txt with 14 columns:
    time (s) | Marker channel | sensor 1 | sensor 2 | ... | sensor 12
Sampling rate: 10 Hz | Pressure units: mmHg

Usage (from project root in WSL / Linux):
    python scripts/download_sparc_hrm.py

What it does:
    1. Creates  data/sparc_hrm/sub-XX/ directories
    2. Downloads each subject's primary .txt file and _params.txt
    3. Prints a summary of subjects downloaded

After running, load a recording with:
    from ens_gi_digital.patient_data import PatientDataLoader
    loader = PatientDataLoader('data')
    time, forces = loader.load_sparc_hrm(subject_id=1, normalize=False)
    # time:   [T] array in ms
    # forces: [T, 12] array in mmHg
"""

import urllib.request
import json
import sys
from pathlib import Path

# Pennsieve Discover API — dataset 33, version 3
PENNSIEVE_API = "https://api.pennsieve.io/discover/datasets/33/versions/3/files"
S3_BASE = "https://sparc-prod-aod-discover-publish50-use1.s3.amazonaws.com/33"
OUT_DIR = Path(__file__).parent.parent / "data" / "sparc_hrm"

# Subject IDs present in the dataset (some are skipped in the original numbering)
SUBJECT_IDS = [
    1, 11, 13, 14, 15, 17, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37,
    38, 39, 41, 43, 46, 48, 49, 50, 51, 52, 53, 54, 56, 57, 58, 59, 62, 63,
]


def fetch_manifest() -> list:
    """Fetch the dataset file manifest from the Pennsieve API."""
    print("Fetching dataset manifest from Pennsieve ...")
    url = f"{PENNSIEVE_API}?limit=500&offset=0"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
        return data.get("files", [])
    except Exception as exc:
        print(f"[WARN] Could not fetch manifest: {exc}")
        return []


def download_file(url: str, dest: Path) -> bool:
    """Download a single file with a simple progress indicator."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return False  # already downloaded

    try:
        def _progress(block_count, block_size, total_size):
            downloaded = block_count * block_size
            if total_size > 0:
                pct = min(downloaded / total_size * 100, 100)
                print(f"\r    {pct:5.1f}%", end="", flush=True)

        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print(f"\r    OK ({dest.stat().st_size // 1024} KB)      ")
        return True
    except Exception as exc:
        print(f"\r    FAILED: {exc}")
        if dest.exists():
            dest.unlink()
        return False


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = fetch_manifest()

    # Build a map: path → download URL from manifest
    path_to_url: dict = {}
    for entry in manifest:
        fpath = entry.get("path", "")
        uri = entry.get("uri", "")
        if fpath and uri:
            path_to_url[fpath] = uri

    downloaded = 0
    skipped = 0
    failed = 0

    for sub_id in SUBJECT_IDS:
        sub_dir = OUT_DIR / f"sub-{sub_id:02d}"
        sub_dir.mkdir(exist_ok=True)

        # Try to find this subject's primary file and params in the manifest
        sub_prefix = f"files/primary/sub-{sub_id:02d}/"
        sub_files = {
            p: u for p, u in path_to_url.items() if p.startswith(sub_prefix)
        }

        if sub_files:
            for path, url in sub_files.items():
                filename = Path(path).name
                dest = sub_dir / filename
                print(f"  sub-{sub_id:02d}/{filename}")
                ok = download_file(url, dest)
                if ok:
                    downloaded += 1
                else:
                    skipped += 1
        else:
            # Manifest unavailable — print manual instructions
            print(
                f"  [INFO] sub-{sub_id:02d}: manifest not available. "
                f"Download manually from https://sparc.science/datasets/33"
            )
            skipped += 1

    print(f"\n{'='*60}")
    print(f"Downloaded: {downloaded} files")
    print(f"Skipped (already present or unavailable): {skipped}")
    print(f"Failed: {failed}")
    print(f"\nData directory: {OUT_DIR}")
    print(f"\nTo load subject 1 HRM:")
    print("  from ens_gi_digital.patient_data import PatientDataLoader")
    print("  loader = PatientDataLoader('data')")
    print("  time, forces = loader.load_sparc_hrm(subject_id=1, normalize=False)")
    print("  # forces: shape (T, 12), pressure in mmHg")
    print("\nNOTE: SPARC requires a free account for bulk download.")
    print("      Alternatively, use the SPARC portal web UI:")
    print("      https://sparc.science/datasets/33/version/3")


if __name__ == "__main__":
    main()
