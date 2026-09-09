#!/usr/bin/env python3
"""Loopback-only annotation server; uses only the Python standard library."""

import argparse
import copy
import hashlib
import secrets
import json
import mimetypes
import os
from pathlib import Path
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse


def valid_label(label):
    if not isinstance(label, dict) or type(label.get("skip", False)) is not bool:
        return False
    score = label.get("score")
    return (isinstance(label.get("reasoning", ""), str)
            and ((type(score) is int and 0 <= score <= 9)
                 or (score is None and label.get("skip") is True)))


def field(value, key):
    for part in key.split("."):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def home_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.home() / path
    return path.resolve(strict=True)


def browse(value):
    home = Path.home().resolve()
    path = home_path(value or ".")
    if not path.is_relative_to(home) or not path.is_dir():
        raise ValueError("Choose a folder within your Home directory.")
    folders, files = [], []
    for child in path.iterdir():
        if child.name.startswith("."):
            continue
        if not child.resolve().is_relative_to(home):
            continue
        if child.is_dir():
            folders.append(child.name)
        elif (
            child.is_file() and child.suffix.lower() == ".json" and child.name != "annotations.json"
        ):
            files.append(child.name)
    metadata_dir = path / "metadata"
    if metadata_dir.is_dir() and metadata_dir.resolve().is_relative_to(home):
        files.extend(
            "metadata/" + child.name
            for child in metadata_dir.iterdir()
            if child.is_file()
            and child.suffix.lower() == ".json"
            and child.resolve().is_relative_to(home)
        )
    return {
        "path": str(path.relative_to(home)),
        "parent": str(path.parent.relative_to(home)) if path != home else None,
        "folders": sorted(folders, key=str.casefold),
        "json_files": sorted(files, key=str.casefold),
    }


class Dataset:
    def __init__(self, config):
        self.config = dict(config)
        self.root = home_path(config["folder"])
        filename = config.get("json_file", "").strip()
        if self.root.name == "metadata":
            self.root = self.root.parent
            filename = (
                "metadata/" + filename
                if filename and not filename.startswith("metadata/")
                else filename
            )
        metadata_dir = self.safe("metadata")
        candidates = (
            list(metadata_dir.glob("*.json"))
            if metadata_dir.is_dir()
            else [p for p in self.root.glob("*.json") if p.name != "annotations.json"]
        )
        if not filename and len(candidates) != 1:
            raise ValueError("Select one metadata JSON file for this batch.")
        if filename and not (self.root / filename).exists() and (metadata_dir / filename).is_file():
            filename = "metadata/" + filename
        source = self.safe(filename) if filename else self.safe(str(candidates[0]))
        if source.name == "annotations.json" or source.is_relative_to(self.safe("annotations")):
            raise ValueError("Select source metadata, not an annotation output.")
        self.source_metadata = str(source.relative_to(self.root))
        self.config.update(folder=str(self.root), json_file=self.source_metadata)
        self.token = secrets.token_hex(16)
        raw = json.loads(source.read_text())
        self.merged_document = None
        if isinstance(raw, dict) and raw.get("artifact_type") == "merged_annotations":
            self.load_merged(source, raw)
            return
        key = config.get("records_key", "").strip()
        if key:
            raw = field(raw, key)
        elif isinstance(raw, dict):
            for name in ("records", "items", "data", "images"):
                if isinstance(raw.get(name), list):
                    raw = raw[name]
                    break
        if isinstance(raw, dict):
            entries = list(raw.items())
        elif isinstance(raw, list):
            entries = list(enumerate(raw))
        else:
            raise ValueError("Metadata must contain an array or an object of records.")
        self.records, self.paths = [], {}
        self.legacy_ids = {}
        for index, record in entries:
            if not isinstance(record, dict):
                record = {"value": record}
            id_key = config.get("id_key", "").strip()
            rid = (
                str(field(record, id_key))
                if id_key
                else str(
                    record.get(
                        "id",
                        record.get(
                            "thread_id",
                            index
                            if isinstance(raw, dict)
                            else "sha256:"
                            + hashlib.sha256(
                                json.dumps(record, sort_keys=True, ensure_ascii=False).encode()
                            ).hexdigest(),
                        ),
                    )
                )
            )
            if rid in self.paths:
                raise ValueError("Duplicate record IDs; select a unique ID field.")
            legacy_id = str(field(record, id_key)) if id_key else str(record.get("id", index))
            if legacy_id in self.legacy_ids:
                raise ValueError("Duplicate legacy IDs; cannot safely match previous annotations.")
            self.legacy_ids[legacy_id] = rid
            image_key = config.get("image_key", "").strip()
            image = (
                field(record, image_key)
                if image_key
                else next(
                    (
                        record[k]
                        for k in (
                            "image_url",
                            "image",
                            "image_path",
                            "file_name",
                            "filename",
                            "path",
                        )
                        if isinstance(record.get(k), str)
                    ),
                    None,
                )
            )
            path = None
            if isinstance(image, str):
                for relative in (image, "images/" + image, "image/" + image):
                    candidate = self.safe(relative)
                    if candidate.is_file():
                        path = candidate
                        break
            self.paths[rid] = path
            from urllib.parse import quote

            self.records.append(
                {
                    "id": rid,
                    "metadata": record,
                    "text": field(record, config["text_key"])
                    if config.get("text_key", "").strip()
                    else None,
                    "image_url": "/image?id=" + quote(rid, safe="") if path else None,
                }
            )
        self.by_id = {record["id"]: record for record in self.records}
        relative_source = (
            source.relative_to(metadata_dir)
            if source.is_relative_to(metadata_dir)
            else source.relative_to(self.root)
        )
        self.output = self.safe(str(Path("annotations") / relative_source))
        legacy = next(
            (
                path
                for path in (
                    self.root / "annotations" / "annotations.json",
                    self.root / "annotations.json",
                )
                if path.exists()
            ),
            None,
        )
        self.notice = (
            f"Legacy backup: {legacy.relative_to(self.root)}. "
            "Current annotations are loaded from and saved to the batch output shown above."
            if legacy
            else ""
        )
        self.annotations = {"pointwise": {}, "pairwise": {}}
        self.saved_records = {}
        self.output_digest = None
        if self.output.exists():
            encoded = self.output.read_bytes()
            self.output_digest = hashlib.sha256(encoded).hexdigest()
            stored = json.loads(encoded)
            if (
                not isinstance(stored, dict)
                or stored.get("schema_version") != 2
                or stored.get("source_metadata") != self.source_metadata
                or not isinstance(stored.get("records"), list)
                or not isinstance(stored.get("pairwise", {}), dict)
            ):
                raise ValueError(
                    "Existing batch annotations have an unsupported format or source; nothing was changed."
                )
            self.annotations["pairwise"] = stored.get("pairwise", {})
            for saved in stored["records"]:
                rid = saved["id"]
                label = saved["annotation"]
                if (
                    rid in self.saved_records
                    or not valid_label(label)
                ):
                    raise ValueError("Existing batch annotations are invalid; nothing was changed.")
                if rid in self.by_id and saved["metadata"] != self.by_id[rid]["metadata"]:
                    raise ValueError(
                        "An already annotated record changed its metadata. Restore it or use a new batch filename to avoid applying stale labels."
                    )
                self.saved_records[rid] = saved
                if rid in self.by_id:
                    self.annotations["pointwise"][rid] = label

    def load_merged(self, source, document):
        """Edit a versioned snapshot in place, addressing duplicate IDs by row."""
        rows = document.get("records")
        if document.get("schema_version") != 2 or not isinstance(rows, list):
            raise ValueError("Invalid merged annotation records.")
        self.merged_document = document
        self.output = source
        self.output_digest = hashlib.sha256(source.read_bytes()).hexdigest()
        self.records, self.paths, self.saved_records = [], {}, {}
        self.annotations = {"pointwise": {}, "pairwise": {}}
        self.legacy_ids = {}
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or not isinstance(row.get("metadata"), dict):
                raise ValueError("Invalid merged record metadata.")
            label = row.get("annotation")
            if label is not None and not valid_label(label):
                raise ValueError("Invalid merged annotation label.")
            rid = str(index)
            image = row.get("image_path")
            path = self.safe(image) if isinstance(image, str) and image else None
            self.paths[rid] = path if path and path.is_file() else None
            self.records.append({"id": rid, "original_id": row.get("id"),
                                 "metadata": row["metadata"],
                                 "image_url": "/image?id=" + rid if self.paths[rid] else None})
            if label is not None:
                self.annotations["pointwise"][rid] = label
        self.by_id = {row["id"]: row for row in self.records}
        legacy_count = len(document.get("legacy_partitions", []))
        self.notice = (
            "Editing this merged version only; original partitions remain unchanged. "
            "Skipped samples are excluded when exporting this version. "
            f"{legacy_count} legacy partitions are retained but are not editable here."
        )

    def import_legacy(self, path):
        """Explicit one-time import for a known batch, retaining the original file.

        Old array-position IDs require the original metadata ordering. Never call
        automatically when a different/new batch is opened.
        """
        legacy = json.loads(Path(path).read_text())
        if not isinstance(legacy, dict) or not isinstance(legacy.get("pointwise"), dict):
            raise ValueError("Unsupported legacy annotations.")
        pending = []
        skipped = 0
        for old_id, label in legacy["pointwise"].items():
            if old_id not in self.legacy_ids:
                raise ValueError("An old annotation cannot be matched; no import was performed.")
            if (
                not isinstance(label, dict)
                or type(label.get("score")) is not int
                or not 0 <= label["score"] <= 9
                or not isinstance(label.get("reasoning", ""), str)
            ):
                raise ValueError("Invalid legacy annotation; no import was performed.")
            rid = self.legacy_ids[old_id]
            if rid in self.saved_records:
                skipped += 1  # Never overwrite a newer annotation during migration.
            else:
                pending.append(
                    {
                        "mode": "pointwise",
                        "id": rid,
                        "score": label["score"],
                        "reasoning": label.get("reasoning", ""),
                    }
                )
        for label in pending:
            self.save(label)
        return {"imported": len(pending), "existing_preserved": skipped}

    def safe(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Paths must stay within the selected folder.")
        return path

    def state(self):
        return {
            "records": self.records,
            "annotations": self.annotations,
            "config": self.config,
            "output_path": str(self.output.relative_to(self.root)),
            "source_metadata": self.source_metadata,
            "notice": self.notice,
            "dataset_token": self.token,
            "merged": self.merged_document is not None,
        }

    def save(self, data, *, provenance=None):
        mode = data.get("mode")
        reasoning = data.get("reasoning", "")
        if not isinstance(reasoning, str):
            raise ValueError("Reasoning must be text.")
        if mode == "pointwise":
            key, score = data.get("id"), data.get("score")
            skip = data.get("skip", False)
            value = {"score": score, "reasoning": reasoning, "skip": skip}
            if key not in self.paths or not valid_label(value):
                raise ValueError("Choose a valid record and score 0–9, or Skip with no score.")
            if provenance is not None:
                value.update(review_required=True, provenance=provenance)
        elif mode == "pairwise":
            if self.merged_document is not None:
                raise ValueError("Merged snapshots support pointwise review only.")
            left, right, choice = data.get("left"), data.get("right"), data.get("choice")
            if (
                left not in self.paths
                or right not in self.paths
                or left == right
                or choice not in ("left", "tie", "right")
            ):
                raise ValueError("Choose two different records and a preference.")
            key = json.dumps([left, right], ensure_ascii=False, separators=(",", ":"))
            value = {"left": left, "right": right, "choice": choice, "reasoning": reasoning}
        else:
            raise ValueError("Unknown annotation mode.")
        updated = {k: dict(v) for k, v in self.annotations.items()}
        updated[mode][key] = value
        saved_records = dict(self.saved_records)
        if mode == "pointwise":
            path = self.paths[key]
            saved_records[key] = {
                "id": key,
                "metadata": self.by_id[key]["metadata"],
                "image_path": str(path.relative_to(self.root)) if path else None,
                "annotation": value,
            }
        document = {
            "schema_version": 2,
            "source_metadata": self.source_metadata,
            "records": list(saved_records.values()),
            "pairwise": updated["pairwise"],
        }
        if self.merged_document is not None:
            document = copy.deepcopy(self.merged_document)
            document["records"][int(key)]["annotation"] = value
        current_digest = (
            hashlib.sha256(self.output.read_bytes()).hexdigest() if self.output.exists() else None
        )
        if current_digest != self.output_digest:
            raise ValueError("This batch was modified elsewhere. Reload before saving.")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.safe(str(self.output))
        temp = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=self.output.parent,
                prefix=".annotations-",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp = handle.name
                json.dump(document, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, self.output)
            if self.merged_document is not None:
                self.merged_document = document
            self.annotations = updated
            self.saved_records = saved_records
            self.output_digest = hashlib.sha256(self.output.read_bytes()).hexdigest()
        finally:
            if temp and os.path.exists(temp):
                os.unlink(temp)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass  # Do not log dataset names, IDs, or contents.

    def reply(self, status, body, content_type="application/json"):
        if content_type == "application/json":
            body = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def trusted(self):
        expected = f"127.0.0.1:{self.server.server_port}"
        return (
            self.headers.get("Host") == expected
            and self.headers.get("Origin", "http://" + expected) == "http://" + expected
        )

    def do_GET(self):
        if not self.trusted():
            return self.reply(403, {"error": "Use the displayed loopback URL."})
        url = urlparse(self.path)
        if url.path == "/":
            return self.reply(
                200, Path(__file__).with_name("index.html").read_bytes(), "text/html; charset=utf-8"
            )
        if url.path == "/api/browse":
            try:
                return self.reply(200, browse(parse_qs(url.query).get("path", ["."])[0]))
            except (OSError, ValueError):
                return self.reply(
                    400,
                    {
                        "error": "Cannot browse this folder. Choose an accessible folder within Home."
                    },
                )
        if url.path == "/api/state":
            return self.reply(
                200,
                self.server.dataset.state()
                if self.server.dataset
                else {"records": [], "annotations": {}},
            )
        if url.path == "/image" and self.server.dataset:
            rid = parse_qs(url.query).get("id", [""])[0]
            path = self.server.dataset.paths.get(rid)
            if path:
                try:
                    path = self.server.dataset.safe(str(path))
                    mime = mimetypes.guess_type(path.name)[0]
                    if mime not in (
                        "image/png",
                        "image/jpeg",
                        "image/webp",
                        "image/gif",
                        "image/bmp",
                        "image/avif",
                    ):
                        return self.reply(400, {"error": "Unsupported image type."})
                    return self.reply(200, path.read_bytes(), mime)
                except (OSError, ValueError):
                    pass
        self.reply(404, {"error": "Not found."})

    def do_POST(self):
        if (
            not self.trusted()
            or self.headers.get("Content-Type", "").split(";")[0] != "application/json"
        ):
            return self.reply(403, {"error": "Same-origin JSON requests required."})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 1024 * 1024:
                raise ValueError("Request is empty or too large.")
            data = json.loads(self.rfile.read(size))
            if self.path == "/api/load":
                dataset = Dataset(data)
                self.server.dataset = dataset
                return self.reply(200, dataset.state())
            if self.path == "/api/save" and self.server.dataset:
                if data.get("dataset_token") != self.server.dataset.token:
                    raise ValueError("The active batch changed. Reload this page before saving.")
                self.server.dataset.save(data)
                return self.reply(200, {"ok": True})
            self.reply(400, {"error": "Load a dataset first."})
        except (ValueError, KeyError, TypeError, OSError, IndexError) as error:
            # Do not expose malformed source text or filesystem exception contents.
            message = (
                str(error)
                if type(error) is ValueError
                else "Could not load or save. Check paths, field mappings, JSON format, and file permissions."
            )
            self.reply(400, {"error": message})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", help="Data root for startup loading")
    parser.add_argument("--annotation-file", help="Merged version relative to data root")
    args = parser.parse_args()
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    if bool(args.data_dir) != bool(args.annotation_file):
        parser.error("--data-dir and --annotation-file must be used together")
    server.dataset = Dataset({"folder": str(Path(args.data_dir).expanduser().resolve()), "json_file": args.annotation_file}) if args.data_dir else None
    print(f"Local annotator: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
