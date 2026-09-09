#!/usr/bin/env python3
"""Build reproducible, image-grouped JSONL splits from local annotation batches."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import tempfile

from training_augmentation import augment_train, validate_augmentation

SPLITS = ("train", "validation", "test")


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_field(record, key):
    for part in key.split("."):
        record = record[int(part)] if isinstance(record, list) else record[part]
    return record


def prepare(config):
    augmentation = validate_augmentation(config.get("augmentation"))
    ratios = config.get("ratios", dict(train=0.8, validation=0.1, test=0.1))
    if (
        not isinstance(ratios, dict)
        or set(ratios) != set(SPLITS)
        or any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in ratios.values()
        )
        or not math.isclose(sum(ratios.values()), 1, abs_tol=1e-9)
    ):
        raise ValueError("ratios must contain train, validation, test; nonnegative and sum to 1.")
    root = Path(config["data_dir"]).expanduser().resolve(strict=True)
    output = Path(config["output_dir"]).expanduser().resolve()
    if output.exists():
        raise FileExistsError("Output already exists. Choose a new output directory.")
    annotation_dir = (root / "annotations").resolve()
    selected = config.get("annotation_file")
    annotation_file = None
    selection_mode = "partition_fallback"
    if selected is not None:
        if not isinstance(selected, str) or not selected.strip():
            raise ValueError("annotation_file must be a nonempty path inside data_dir.")
        selection_mode = "explicit"
        annotation_file = (root / Path(selected).expanduser()).resolve(strict=True)
        if not annotation_file.is_relative_to(root) or not annotation_file.is_file():
            raise ValueError("annotation_file must be a file inside data_dir.")
    else:
        merged_dir = root / "merged_annotations"
        # Fixed-width UTC timestamps sort chronologically; mtime is intentionally ignored.
        versions = sorted(
            path for path in merged_dir.glob("merged_annotations_*.json")
            if re.fullmatch(r"merged_annotations_\d{8}T\d{12}Z_[0-9a-f]{8}\.json", path.name)
        )
        legacy = merged_dir / "merged_annotations.json"
        if versions:
            annotation_file = versions[-1]
            selection_mode = "latest_merged"
        elif legacy.exists():
            annotation_file = legacy
            selection_mode = "legacy_merged"
        if annotation_file is not None:
            annotation_file = annotation_file.resolve(strict=True)
            if not annotation_file.is_relative_to(root) or not annotation_file.is_file():
                raise ValueError("Selected merged annotation must be a file inside data_dir.")
        elif not annotation_dir.is_dir():
            raise ValueError("annotations directory does not exist.")
    selection = {
        "mode": selection_mode,
        "path": str(annotation_file if annotation_file is not None else annotation_dir),
    }
    if not annotation_dir.is_relative_to(root):
        raise ValueError("annotations directory must be inside data_dir.")
    if (
        output.is_relative_to(annotation_dir)
        or output.is_relative_to((root / "images").resolve())
        or (annotation_file is not None and annotation_file.parent != root
            and output.is_relative_to(annotation_file.parent))
    ):
        raise ValueError("Output must not be inside annotation input directories or images/.")
    text_key = config.get("text_key", "text")
    if not isinstance(text_key, str) or not text_key:
        raise ValueError("Specify text_key, the metadata field actually used as model input.")
    seed = config.get("seed", 42)
    if type(seed) is not int:
        raise ValueError("seed must be an integer.")
    task = config.get("task", "pointwise-score-v1")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a nonempty scoring-task/version identifier.")
    include_reasoning = config.get("include_reasoning", True)
    allow_empty_text = config.get("allow_empty_text", False)
    candidates = defaultdict(list)
    rejected, conflicts, skipped, inventory = [], [], [], []
    hash_cache = {}
    input_records = 0
    skipped_unscored = 0
    skipped_unreviewed = 0
    skipped_explicit = 0
    batches = [annotation_file] if annotation_file is not None else sorted(annotation_dir.rglob("*.json"))
    for batch in batches:
        relative_batch = str(batch.relative_to(root))
        if annotation_file is None and any(part.startswith("merged_") for part in batch.relative_to(annotation_dir).parts):
            skipped.append({"file": relative_batch, "reason": "merged_annotations"})
            continue
        if not batch.resolve().is_relative_to(root):
            raise ValueError("Annotation symlinks must stay within data_dir.")
        # A corrupt file aborts rather than silently dropping an entire batch.
        try:
            document = json.loads(batch.read_text())
        except (ValueError, OSError) as error:
            raise ValueError(
                "Cannot parse an annotation file; check it locally before retrying."
            ) from error
        if annotation_file is None and isinstance(document, dict) and document.get("artifact_type") == "merged_annotations":
            skipped.append({"file": relative_batch, "reason": "merged_annotations"})
            continue
        if selection_mode in ("latest_merged", "legacy_merged") and (
            not isinstance(document, dict)
            or type(document.get("schema_version")) is not int
            or document.get("schema_version") != 2
            or document.get("artifact_type") != "merged_annotations"
            or not isinstance(document.get("records"), list)
        ):
            raise ValueError("Selected merged annotation must be a version 2 merged_annotations artifact with records.")
        inventory.append({"path": relative_batch, "sha256": digest_file(batch)})
        if not isinstance(document, dict) or document.get("schema_version") != 2:
            skipped.append({"file": relative_batch, "reason": "legacy_or_unsupported_schema"})
            continue
        if not isinstance(document.get("records"), list):
            raise ValueError("A version 2 annotation file has no records array.")
        for number, record in enumerate(document["records"]):
            input_records += 1
            source = {"file": relative_batch, "record_index": number}
            original_sources = document.get("record_sources")
            if isinstance(original_sources, list) and number < len(original_sources):
                source["original_source"] = original_sources[number]
            try:
                if not isinstance(record, dict):
                    raise ValueError("invalid_record")
                source["record_id"] = record.get("id")
                annotation = record.get("annotation")
                if annotation is None:
                    skipped_unscored += 1
                    continue
                if not isinstance(annotation, dict):
                    raise ValueError("invalid_annotation")
                explicit_skip = annotation.get("skip", False)
                if type(explicit_skip) is not bool:
                    raise ValueError("invalid_skip")
                if explicit_skip:
                    skipped_explicit += 1
                    continue
                if annotation.get("review_required") is True:
                    skipped_unreviewed += 1
                    continue
                score = annotation.get("score")
                if score is None or (isinstance(score, str) and not score.strip()):
                    skipped_unscored += 1
                    continue
                if type(score) is not int or not 0 <= score <= 9:
                    raise ValueError("invalid_score")
                reasoning = annotation.get("reasoning", "")
                if not isinstance(reasoning, str):
                    raise ValueError("invalid_reasoning")
                try:
                    text = get_field(record["metadata"], text_key)
                except (KeyError, IndexError, TypeError, ValueError):
                    if not allow_empty_text:
                        raise ValueError("missing_text")
                    text = ""
                if not isinstance(text, str):
                    raise ValueError("invalid_text")
                text = text.strip()
                if not text and not allow_empty_text:
                    raise ValueError("empty_text")
                image = record.get("image_path")
                if not isinstance(image, str) or not image:
                    raise ValueError("missing_image_path")
                path = (root / image).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("image_outside_data_dir")
                if not path.is_file():
                    raise ValueError("missing_image")
                if path.suffix.lower() not in (
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".webp",
                    ".gif",
                    ".bmp",
                    ".avif",
                ):
                    raise ValueError("unsupported_image_extension")
                if path.stat().st_size == 0:
                    raise ValueError("empty_image")
                if path not in hash_cache:
                    hash_cache[path] = digest_file(path)
                image_hash = hash_cache[path]
                key = hashlib.sha256(
                    json.dumps([task, image_hash, text], ensure_ascii=False).encode()
                ).hexdigest()
                candidates[key].append(
                    {
                        "id": key,
                        "image_sha256": image_hash,
                        "image": os.path.relpath(path, output),
                        "text": text,
                        "score": score,
                        "reasoning": reasoning,
                        "source": source,
                    }
                )
            except (KeyError, TypeError, ValueError, OSError) as error:
                reason = (
                    str(error)
                    if type(error) is ValueError
                    else "invalid_record_or_unreadable_image"
                )
                rejected.append({"source": source, "reason": reason})
    samples = []
    duplicates_merged = 0
    for key, rows in sorted(candidates.items()):
        if len({row["score"] for row in rows}) > 1:
            conflicts.append({"id": key, "reason": "conflicting_scores", "candidates": rows})
            continue
        first = rows[0]
        sample = {k: first[k] for k in ("id", "image_sha256", "image", "text", "score")}
        sample["sources"] = [row["source"] for row in rows]
        if include_reasoning:
            # Preserve all rationale alternatives; pick a deterministic nonempty primary.
            alternatives = sorted({row["reasoning"] for row in rows if row["reasoning"]})
            sample["reasoning"] = alternatives[0] if alternatives else ""
            sample["reasoning_alternatives"] = alternatives
        samples.append(sample)
        duplicates_merged += len(rows) - 1
    groups = defaultdict(list)
    for sample in samples:
        groups[sample["image_sha256"]].append(sample)
    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    # Larger image groups first, seeded order breaks ties. Fractions target sample counts.
    keys.sort(key=lambda key: -len(groups[key]))
    splits = {split: [] for split in SPLITS}
    targets = {split: len(samples) * ratios[split] for split in SPLITS}
    for key in keys:
        eligible = [split for split in SPLITS if ratios[split] > 0]
        split = max(eligible, key=lambda name: targets[name] - len(splits[name]))
        splits[split].extend(groups[key])
    for rows in splits.values():
        rng.shuffle(rows)
    counts = {split: len(rows) for split, rows in splits.items()}
    report = {
        "schema_version": 1,
        "config": {**config, "ratios": ratios, "seed": seed, "text_key": text_key, "task": task},
        "annotation_selection": selection,
        "input_records": input_records,
        "skipped_unscored": skipped_unscored,
        "skipped_unreviewed": skipped_unreviewed,
        "skipped_explicit": skipped_explicit,
        "unique_samples": len(samples),
        "duplicates_merged": duplicates_merged,
        "conflict_groups": len(conflicts),
        "rejected_records": len(rejected),
        "image_groups": len(groups),
        "counts": counts,
        "actual_ratios": {k: v / len(samples) if samples else 0 for k, v in counts.items()},
        "skipped_files": skipped,
        "input_files": inventory,
        "warnings": [
            "Ratios are approximate because image groups are indivisible.",
            "Image bytes are hashed; image decoding and near-duplicate detection are not performed.",
            "Rebuilding with additional inputs can change split membership; keep a versioned release for a fixed evaluation set.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".training-build-", dir=output.parent))
    try:
        original_train = list(splits["train"])
        if augmentation["enabled"]:
            reserved = {
                row["image_sha256"] for split in ("validation", "test") for row in splits[split]
            }
            variants, augmentation_stats = augment_train(
                original_train, output, staging, augmentation, seed, reserved
            )
            splits["train"] = original_train + variants
            report["augmentation"] = augmentation_stats
            report["augmentation"]["config"] = augmentation
            report["exported_counts"] = {name: len(rows) for name, rows in splits.items()}
            report["warnings"].append(
                "Augmented labels are inherited assumptions, not new human judgments. Counts/actual_ratios describe originals; exported_counts includes train augmentation."
            )
            report["warnings"].append(
                "all.jsonl contains deduplicated originals only; augmented rows appear only in train.jsonl."
            )
        exports = {"train_original": original_train} if augmentation["enabled"] else {}
        for name, rows in {
            **exports,
            "all": samples,
            **splits,
            "conflicts": conflicts,
            "rejected": rejected,
        }.items():
            with (staging / (name + ".jsonl")).open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        (staging / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        if output.exists():
            raise FileExistsError("Output appeared during processing; refusing to overwrite it.")
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON config; CLI options override it")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--annotation-file", help="Export only this JSON file; relative paths use data-dir")
    parser.add_argument("--text-key")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--train-ratio", type=float)
    parser.add_argument("--validation-ratio", type=float)
    parser.add_argument("--test-ratio", type=float)
    args = parser.parse_args()
    config = json.loads(Path(args.config).expanduser().read_text()) if args.config else {}
    for key in ("data_dir", "output_dir", "annotation_file", "text_key", "seed"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    ratios = dict(config.get("ratios", dict(train=0.8, validation=0.1, test=0.1)))
    for split in SPLITS:
        value = getattr(args, split + "_ratio")
        if value is not None:
            ratios[split] = value
    config["ratios"] = ratios
    try:
        report = prepare(config)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(
            2,
            f"Build failed ({type(error).__name__}). Check config and local inputs; no private record contents are printed.\n",
        )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "annotation_selection",
                    "input_records",
                    "skipped_unscored",
                    "skipped_unreviewed",
                    "skipped_explicit",
                    "unique_samples",
                    "duplicates_merged",
                    "conflict_groups",
                    "rejected_records",
                    "counts",
                    "exported_counts",
                    "augmentation",
                )
                if key in report
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
