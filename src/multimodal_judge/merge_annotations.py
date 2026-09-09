"""Merge annotation partitions without consuming previous merge outputs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


PREFIX = "merged_"


def merge_annotations(annotation_dir: Path, output: Path) -> dict:
    """Preserve records verbatim, with parallel provenance and legacy documents.

    This is a lossless merge, not training-data preparation: repeated IDs and
    review_required flags are retained. No score selection or deduplication occurs.
    """
    annotation_dir = annotation_dir.resolve(strict=True)
    output = output.resolve()
    if not output.name.startswith(PREFIX) or output.suffix != ".json":
        raise ValueError("Output must be named merged_*.json")
    if output.is_relative_to(annotation_dir):
        raise ValueError("Output must be outside the annotation input directory")
    sources, records, provenance, legacy, pairwise = [], [], [], [], []
    for path in sorted(annotation_dir.rglob("*.json")):
        if any(part.startswith(PREFIX) for part in path.relative_to(annotation_dir).parts):
            continue
        if not path.resolve().is_relative_to(annotation_dir):
            raise ValueError(f"Input symlink escapes annotation directory: {path.name}")
        raw = path.read_bytes()
        document = json.loads(raw)
        relative = path.relative_to(annotation_dir).as_posix()
        if not isinstance(document, dict):
            raise ValueError(f"Expected an annotation object: {relative}")
        # Also recognize renamed merge artifacts by their explicit marker.
        if document.get("artifact_type") == "merged_annotations":
            continue
        sources.append({"file": relative, "sha256": hashlib.sha256(raw).hexdigest(),
                        "source_metadata": document.get("source_metadata")})
        if document.get("schema_version") == 2:
            rows = document.get("records")
            if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
                raise ValueError(f"Invalid records array: {relative}")
            records.extend(rows)
            provenance.extend({"file": relative, "record_index": i} for i in range(len(rows)))
            pairwise.append({"file": relative, "annotations": document.get("pairwise", {})})
        elif "pointwise" in document and "pairwise" in document:
            legacy.append({"file": relative, "document": document})
        else:
            raise ValueError(f"Unsupported annotation schema: {relative}")
    if not sources:
        raise ValueError("No annotation input files found")
    result = {"artifact_type": "merged_annotations", "schema_version": 2,
              "source_metadata": None, "records": records, "record_sources": provenance,
              "pairwise": {}, "pairwise_partitions": pairwise,
              "legacy_partitions": legacy, "sources": sources}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Validate every input before atomically replacing the previous result.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", type=Path, default=Path("data/annotations"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/merged_annotations/merged_annotations.json"))
    args = parser.parse_args()
    try:
        result = merge_annotations(args.annotation_dir, args.output)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Merge failed: {error}\n")
    print(f"Merged {len(result['sources'])} files: {len(result['records'])} records, "
          f"{len(result['legacy_partitions'])} legacy partitions -> {args.output}")


if __name__ == "__main__":
    main()
