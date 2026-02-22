"""Download and unpack the Zenodo 3-channel EGG dataset.

Dataset: Popovic et al. (2020) — Three-channel surface EGG during fasting
         and post-prandial states in 20 healthy individuals.
DOI:     https://doi.org/10.5281/zenodo.3878435
License: CC BY 4.0

Usage (from project root in WSL / Linux):
    python scripts/download_zenodo_egg.py

What it does:
    1. Creates  data/zenodo_egg/
    2. Downloads EGG-database.zip (~613 KB)
    3. Unzips all 40 recordings (ID1_fasting.txt … ID20_postprandial.txt)
    4. Verifies the expected file count

After running, load a recording with:
    from ens_gi_digital.patient_data import PatientDataLoader
    loader = PatientDataLoader('data')
    time, voltages = loader.load_zenodo_egg(subject_id=1, condition='fasting')
"""

import urllib.request
import zipfile
import sys
from pathlib import Path

ZENODO_ZIP_URL = "https://zenodo.org/record/3878435/files/EGG-database.zip"
OUT_DIR = Path(__file__).parent.parent / "data" / "zenodo_egg"
EXPECTED_FILES = 40  # 20 subjects × 2 conditions


def download(url: str, dest: Path) -> None:
    print(f"Downloading {url}")
    print(f"  -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _progress(block_count, block_size, total_size):
        downloaded = block_count * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            bar = "#" * int(pct / 5)
            print(f"\r  [{bar:<20}] {pct:5.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()  # newline after progress bar


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = OUT_DIR / "EGG-database.zip"

    # Download
    if zip_path.exists():
        print(f"[SKIP] ZIP already exists: {zip_path}")
    else:
        download(ZENODO_ZIP_URL, zip_path)

    # Unzip
    print(f"Extracting to {OUT_DIR} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Extract only .txt files, stripping any subdirectory prefix
        for member in zf.namelist():
            if member.endswith(".txt"):
                filename = Path(member).name
                dest_file = OUT_DIR / filename
                if dest_file.exists():
                    continue
                dest_file.write_bytes(zf.read(member))
                print(f"  + {filename}")

    # Verify
    txt_files = list(OUT_DIR.glob("*.txt"))
    print(f"\n[OK] {len(txt_files)} .txt files in {OUT_DIR}")

    if len(txt_files) < EXPECTED_FILES:
        print(
            f"[WARN] Expected {EXPECTED_FILES} files, found {len(txt_files)}. "
            "Check the ZIP contents."
        )
        sys.exit(1)
    else:
        print(f"[OK] All {EXPECTED_FILES} recordings present.")
        print(f"\nTo load subject 1 fasting EGG:")
        print("  from ens_gi_digital.patient_data import PatientDataLoader")
        print("  loader = PatientDataLoader('data')")
        print("  time, voltages = loader.load_zenodo_egg(1, 'fasting')")
        print(f"  # time: {2400} samples, ~20 min @ 2 Hz")
        print(f"  # voltages: shape (2400, 3)")


if __name__ == "__main__":
    main()
