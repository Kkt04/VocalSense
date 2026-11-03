"""
RAVDESS Dataset Parser and Manifest Generator.
Handles dataset downloading, filename parsing, metadata extraction,
and data integrity verification for speech emotion recognition.
"""

import os
import re
import zipfile
import urllib.request
from typing import Dict, Any, List, Optional
import pandas as pd
import soundfile as sf

# RAVDESS Official Emotion ID Mapping
EMOTION_MAP = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}

INTENSITY_MAP = {
    1: "normal",
    2: "strong",
}

STATEMENT_MAP = {
    1: "Kids are talking by the door",
    2: "Dogs are sitting by the door",
}

MODALITY_MAP = {
    1: "full-AV",
    2: "video-only",
    3: "audio-only",
}

VOCAL_CHANNEL_MAP = {
    1: "speech",
    2: "song",
}

ZENODO_RAVDESS_URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip"


def parse_ravdess_filename(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Parse a RAVDESS audio filename according to the official specification.

    Filename format:
    Modality(01-03)-Channel(01-02)-Emotion(01-08)-Intensity(01-02)-Statement(01-02)-Repetition(01-02)-Actor(01-24).wav
    Example: 03-01-05-02-01-01-12.wav
    """
    basename = os.path.basename(file_path)
    stem, ext = os.path.splitext(basename)

    if ext.lower() != ".wav":
        return None

    parts = stem.split("-")
    if len(parts) != 7:
        return None

    try:
        modality_id = int(parts[0])
        vocal_channel_id = int(parts[1])
        emotion_id = int(parts[2])
        intensity_id = int(parts[3])
        statement_id = int(parts[4])
        repetition_id = int(parts[5])
        actor_id = int(parts[6])
    except ValueError:
        return None

    if emotion_id not in EMOTION_MAP:
        return None

    gender = "female" if (actor_id % 2 == 0) else "male"

    return {
        "file_path": os.path.abspath(file_path),
        "filename": basename,
        "modality_id": modality_id,
        "modality": MODALITY_MAP.get(modality_id, "unknown"),
        "vocal_channel_id": vocal_channel_id,
        "vocal_channel": VOCAL_CHANNEL_MAP.get(vocal_channel_id, "unknown"),
        "emotion_id": emotion_id,
        "emotion_label": EMOTION_MAP[emotion_id],
        "intensity_id": intensity_id,
        "intensity": INTENSITY_MAP.get(intensity_id, "normal"),
        "statement_id": statement_id,
        "statement": STATEMENT_MAP.get(statement_id, "unknown"),
        "repetition_id": repetition_id,
        "actor_id": actor_id,
        "gender": gender,
    }


def download_and_extract_ravdess(
    destination_dir: str = "data/ravdess",
    url: str = ZENODO_RAVDESS_URL,
) -> str:
    """
    Download and extract the official RAVDESS Audio Speech zip file from Zenodo.
    """
    os.makedirs(destination_dir, exist_ok=True)
    zip_path = os.path.join(destination_dir, "Audio_Speech_Actors_01-24.zip")

    # Check if files are already extracted
    actor_dirs = [d for d in os.listdir(destination_dir) if d.startswith("Actor_")]
    if len(actor_dirs) >= 24:
        print(f"RAVDESS dataset already extracted in {destination_dir} (found {len(actor_dirs)} actor directories).")
        return destination_dir

    if not os.path.exists(zip_path):
        print(f"Downloading RAVDESS dataset from {url}...")
        print(f"Saving to {zip_path} (~198 MB)...")
        import subprocess
        # Use curl directly to leverage system certificates on macOS/Linux
        subprocess.run(["curl", "-L", "-o", zip_path, url], check=True)
        print("\nDownload complete!")

    print(f"Extracting {zip_path} to {destination_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(destination_dir)
    print("Extraction complete!")

    return destination_dir


def build_ravdess_manifest(
    ravdess_dir: str = "data/ravdess",
    output_csv: Optional[str] = "data/ravdess/metadata.csv",
) -> pd.DataFrame:
    """
    Scan the RAVDESS directory, verify each audio file with soundfile,
    and build a structured manifest CSV.
    """
    records: List[Dict[str, Any]] = []

    for root, _, files in os.walk(ravdess_dir):
        for f in sorted(files):
            if f.endswith(".wav"):
                full_path = os.path.join(root, f)
                parsed = parse_ravdess_filename(full_path)
                if parsed:
                    # Check audio file integrity and duration
                    try:
                        info = sf.info(full_path)
                        parsed["duration_sec"] = round(info.duration, 3)
                        parsed["sample_rate"] = info.samplerate
                        parsed["channels"] = info.channels
                        parsed["is_corrupted"] = False
                    except Exception as e:
                        parsed["duration_sec"] = 0.0
                        parsed["sample_rate"] = 0
                        parsed["channels"] = 0
                        parsed["is_corrupted"] = True
                        parsed["error"] = str(e)

                    records.append(parsed)

    df = pd.DataFrame(records)

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Manifest written to {output_csv} with {len(df)} recordings.")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAVDESS Dataset Downloader and Manifest Builder")
    parser.add_argument("--download", action="store_true", help="Download dataset from Zenodo")
    parser.add_argument("--dir", default="data/ravdess", help="Directory for RAVDESS data")
    args = parser.parse_args()

    if args.download:
        download_and_extract_ravdess(args.dir)

    df = build_ravdess_manifest(args.dir)
    print("\nDataset Summary:")
    print(f"Total recordings found: {len(df)}")
    if not df.empty:
        print("\nClass distribution:")
        print(df["emotion_label"].value_counts())
        print("\nActor distribution:")
        print(f"Unique actors: {df['actor_id'].nunique()}")
        print(f"Gender split:\n{df['gender'].value_counts()}")
        corrupted = df[df["is_corrupted"] == True]
        print(f"Corrupted files: {len(corrupted)}")
