#!/usr/bin/env python3
"""Extract only the three newly downloaded LibriTTS-R parquet shards.

The samples are written to both:

1. the original clean-WAV directory used by the existing dataset; and
2. a separate directory under ``codex/new_samples`` for incremental
   degradation processing.

This script intentionally does not apply the LibriTTS-R failed-sample list.
The requested workflow is to generate noisy audio first and filter all old
and new clean/noisy pairs together afterwards.
"""

import csv
import os
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm


PARQUET_DIR = Path("/yangliusha02/datasets/libritts_r/data/train.clean.360")
PARQUET_FILES = [
    PARQUET_DIR / "train.clean.360-00052-of-00062.parquet",
    PARQUET_DIR / "train.clean.360-00055-of-00062.parquet",
    PARQUET_DIR / "train.clean.360-00061-of-00062.parquet",
]

OLD_CLEAN_DIR = Path(
    "/yangliusha02/datasets/libritts_r/data/train_clean_360_wav"
)
NEW_CLEAN_DIR = Path(
    "/yangliusha02/datasets/libritts_r/codex/new_samples/clean"
)
NEW_SCP_PATH = Path(
    "/yangliusha02/datasets/libritts_r/codex/new_samples/new_speech.scp"
)
NEW_METADATA_PATH = Path(
    "/yangliusha02/datasets/libritts_r/codex/new_samples/new_metadata.csv"
)
NEW_IDS_PATH = Path(
    "/yangliusha02/datasets/libritts_r/codex/new_samples/new_ids.txt"
)

BATCH_SIZE = 128


def clean_string(value):
    if value is None:
        return ""
    return str(value).replace("\n", " ").replace("\r", " ").strip()


def get_filename(path_value, audio_info, fallback_id):
    filename = path_value or audio_info.get("path") or f"{fallback_id}.wav"
    filename = os.path.basename(str(filename))
    if not filename.lower().endswith(".wav"):
        filename += ".wav"
    return filename


def main():
    for parquet_path in PARQUET_FILES:
        if not parquet_path.is_file():
            raise FileNotFoundError(f"Missing parquet file: {parquet_path}")

    OLD_CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    NEW_CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    NEW_SCP_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_rows = sum(
        pq.ParquetFile(path).metadata.num_rows for path in PARQUET_FILES
    )
    print(f"Parquet shards: {len(PARQUET_FILES)}")
    print(f"Expected rows: {total_rows}")
    print(f"Existing clean output: {OLD_CLEAN_DIR}")
    print(f"New-only clean output: {NEW_CLEAN_DIR}")

    success = 0
    failed = 0
    overwritten_old = 0
    seen_ids = set()

    with (
        NEW_SCP_PATH.open("w", encoding="utf-8") as scp_file,
        NEW_METADATA_PATH.open("w", encoding="utf-8", newline="") as metadata_file,
        NEW_IDS_PATH.open("w", encoding="utf-8") as ids_file,
    ):
        metadata_writer = csv.writer(metadata_file)
        metadata_writer.writerow(
            [
                "id",
                "path",
                "text_normalized",
                "text_original",
                "speaker_id",
                "chapter_id",
            ]
        )

        with tqdm(total=total_rows, desc="Extracting new shards") as progress:
            for parquet_path in PARQUET_FILES:
                parquet_file = pq.ParquetFile(parquet_path)
                for batch in parquet_file.iter_batches(
                    batch_size=BATCH_SIZE,
                    columns=[
                        "audio",
                        "path",
                        "id",
                        "text_normalized",
                        "text_original",
                        "speaker_id",
                        "chapter_id",
                    ],
                ):
                    data = batch.to_pydict()

                    for index, audio_info in enumerate(data["audio"]):
                        try:
                            if audio_info is None:
                                raise ValueError("audio field is None")

                            audio_bytes = audio_info.get("bytes")
                            if audio_bytes is None:
                                raise ValueError("audio bytes are missing")

                            row_id = clean_string(data["id"][index])
                            filename = get_filename(
                                data["path"][index], audio_info, row_id
                            )
                            utt_id = os.path.splitext(filename)[0]

                            if utt_id in seen_ids:
                                raise ValueError(f"duplicate utterance id: {utt_id}")
                            seen_ids.add(utt_id)

                            new_path = NEW_CLEAN_DIR / filename
                            old_path = OLD_CLEAN_DIR / filename

                            if old_path.exists():
                                overwritten_old += 1

                            # The source parquet contains complete WAV bytes.
                            new_path.write_bytes(audio_bytes)
                            old_path.write_bytes(audio_bytes)

                            text_normalized = clean_string(
                                data["text_normalized"][index]
                            )
                            text_original = clean_string(data["text_original"][index])
                            speaker_id = clean_string(data["speaker_id"][index])
                            chapter_id = clean_string(data["chapter_id"][index])

                            scp_file.write(f"{utt_id} {new_path}\n")
                            ids_file.write(f"{utt_id}\n")
                            metadata_writer.writerow(
                                [
                                    utt_id,
                                    str(new_path),
                                    text_normalized,
                                    text_original,
                                    speaker_id,
                                    chapter_id,
                                ]
                            )
                            success += 1
                        except Exception as error:
                            failed += 1
                            print(
                                f"\n[ERROR] {parquet_path.name}, row={index}: {error}"
                            )
                        finally:
                            progress.update(1)

    print("\n==============================")
    print("Incremental extraction complete")
    print("==============================")
    print(f"Expected rows       : {total_rows}")
    print(f"Successfully written : {success}")
    print(f"Failed rows         : {failed}")
    print(f"Overwritten old WAV : {overwritten_old}")
    print(f"New speech.scp      : {NEW_SCP_PATH}")
    print(f"New metadata        : {NEW_METADATA_PATH}")
    print(f"New IDs             : {NEW_IDS_PATH}")

    if success != total_rows or failed != 0:
        raise RuntimeError(
            f"Extraction count mismatch: success={success}, "
            f"failed={failed}, expected={total_rows}"
        )


if __name__ == "__main__":
    main()
