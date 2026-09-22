#!/usr/bin/env python3
"""Merge old and newly generated LibriTTS-R pairs and filter bad samples.

The script does not delete or move any audio.  It creates a new pipe-delimited
metadata.csv containing absolute paths to the existing clean/noisy files.
"""

import csv
import json
import os
import tarfile
from pathlib import Path


LIBRITTS_ROOT = Path("/yangliusha02/datasets/libritts_r")
OLD_METADATA = LIBRITTS_ROOT / "metadata.csv"
NEW_METADATA = LIBRITTS_ROOT / "codex/new_samples/new_metadata.csv"
FAILED_ARCHIVE = (
    LIBRITTS_ROOT
    / "libritts_r_failed_speech_restoration_examples.tar.gz"
)
NEW_MIXED_CLEAN = LIBRITTS_ROOT / "codex/new_samples/mixed/clean"
NEW_MIXED_NOISY = LIBRITTS_ROOT / "codex/new_samples/mixed/noisy"

OUTPUT_DIR = LIBRITTS_ROOT / "codex/merged_filtered"
OUTPUT_METADATA = OUTPUT_DIR / "metadata.csv"
OUTPUT_IDS = OUTPUT_DIR / "kept_ids.txt"
OUTPUT_REMOVED = OUTPUT_DIR / "removed_failed_ids.txt"
OUTPUT_REPORT = OUTPUT_DIR / "merge_report.json"

FAILED_MEMBER = (
    "./libritts_r_failed_speech_restoration_examples/"
    "train-clean-360_bad_sample_list.txt"
)


def utterance_id(path_value):
    return Path(str(path_value).strip()).stem


def load_failed_ids():
    with tarfile.open(FAILED_ARCHIVE, mode="r:gz") as archive:
        member = archive.extractfile(FAILED_MEMBER)
        if member is None:
            raise FileNotFoundError(f"Missing archive member: {FAILED_MEMBER}")

        failed = set()
        for raw_line in member.read().decode("utf-8", "ignore").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            failed.add(utterance_id(line))
    return failed


def add_record(records, record, source, counters):
    uid = utterance_id(record["audio_path"])

    if uid in records:
        counters["duplicate_ids"] += 1
        counters["duplicate_examples"].append(uid)
        return

    if not os.path.isfile(record["audio_path"]):
        counters["missing_clean"] += 1
        return

    if not os.path.isfile(record["control_path"]):
        counters["missing_noisy"] += 1
        return

    records[uid] = {
        "audio_path": record["audio_path"],
        "text": record["text"],
        "control_path": record["control_path"],
        "source": source,
    }


def main():
    for path in [OLD_METADATA, NEW_METADATA, FAILED_ARCHIVE]:
        if not path.is_file():
            raise FileNotFoundError(path)

    if not NEW_MIXED_CLEAN.is_dir() or not NEW_MIXED_NOISY.is_dir():
        raise FileNotFoundError("New mixed clean/noisy directories are missing")

    failed_ids = load_failed_ids()
    records = {}
    removed_failed = []
    counters = {
        "old_rows": 0,
        "new_rows": 0,
        "old_failed_removed": 0,
        "new_failed_removed": 0,
        "missing_clean": 0,
        "missing_noisy": 0,
        "duplicate_ids": 0,
        "duplicate_examples": [],
    }

    with OLD_METADATA.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter="|")
        for row in reader:
            counters["old_rows"] += 1
            uid = utterance_id(row["audio_path"])
            if uid in failed_ids:
                counters["old_failed_removed"] += 1
                removed_failed.append(uid)
                continue
            add_record(
                records,
                {
                    "audio_path": row["audio_path"].strip(),
                    "text": row["text"].strip(),
                    "control_path": row["control_path"].strip(),
                },
                "old",
                counters,
            )

    with NEW_METADATA.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            counters["new_rows"] += 1
            uid = utterance_id(row["path"])
            if uid in failed_ids:
                counters["new_failed_removed"] += 1
                removed_failed.append(uid)
                continue

            filename = f"{uid}.wav"
            add_record(
                records,
                {
                    "audio_path": str(NEW_MIXED_CLEAN / filename),
                    "text": row["text_normalized"].strip(),
                    "control_path": str(NEW_MIXED_NOISY / filename),
                },
                "new",
                counters,
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ordered_records = [records[uid] for uid in sorted(records)]

    with OUTPUT_METADATA.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["audio_path", "text", "control_path"],
            delimiter="|",
        )
        writer.writeheader()
        writer.writerows(
            {
                "audio_path": item["audio_path"],
                "text": item["text"],
                "control_path": item["control_path"],
            }
            for item in ordered_records
        )

    with OUTPUT_IDS.open("w", encoding="utf-8") as file:
        file.write("\n".join(sorted(records)))
        file.write("\n")

    with OUTPUT_REMOVED.open("w", encoding="utf-8") as file:
        file.write("\n".join(sorted(set(removed_failed))))
        file.write("\n")

    report = {
        "failed_ids_in_official_train_clean_360_list": len(failed_ids),
        "old_rows": counters["old_rows"],
        "new_rows": counters["new_rows"],
        "old_failed_removed": counters["old_failed_removed"],
        "new_failed_removed": counters["new_failed_removed"],
        "removed_failed_total": len(set(removed_failed)),
        "kept_total": len(ordered_records),
        "kept_old": sum(x["source"] == "old" for x in ordered_records),
        "kept_new": sum(x["source"] == "new" for x in ordered_records),
        "missing_clean": counters["missing_clean"],
        "missing_noisy": counters["missing_noisy"],
        "duplicate_ids": counters["duplicate_ids"],
        "duplicate_examples": counters["duplicate_examples"][:20],
        "metadata": str(OUTPUT_METADATA),
    }
    with OUTPUT_REPORT.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if len(ordered_records) != 112675:
        raise RuntimeError(
            f"Unexpected kept count: {len(ordered_records)}; expected 112675"
        )
    if counters["missing_clean"] or counters["missing_noisy"]:
        raise RuntimeError("Some clean/noisy pairs are missing")
    if counters["duplicate_ids"]:
        raise RuntimeError("Duplicate utterance IDs were found")


if __name__ == "__main__":
    main()
