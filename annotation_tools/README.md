# Annotation review and training export

These standalone local tools use the existing project environment. The server
binds only to `127.0.0.1`; datasets remain under the ignored `data/` directory.

## Create and review a version

```sh
uv run python -m multimodal_judge.merge_annotations
uv run python annotation_tools/annotator/server.py --port 8765 \
  --data-dir data \
  --annotation-file merged_annotations/merged_annotations_TIMESTAMP_ID.json
```

Replace the example filename with the version printed by the merge command. Each
normal merge creates a new UTC timestamp and random-suffix filename. An explicit
merge `--output` replaces that named file, so do not target a version under review.
Open `http://127.0.0.1:8765`. The page shows the exact active version and save path.
Alternatively start without dataset arguments and load a metadata partition or a
merged file through the page, with `data/` as the folder.

Click **Skip & next** once to save the current inputs, persist `annotation.skip: true`,
and advance. On the final record it saves and stays on that record.
A skipped record needs no score; existing score/reasoning are retained in the form.
Return to it, provide a score if needed, and click **Save & next** to restore
eligibility. **Skip & next** sets Skip; **Save & next** clears it. Save failures keep the current record and edits.
Missing `skip` fields in older labels mean false. Saving an ordinary annotation is
human confirmation and clears an AI draft's pending-review flag, as before.

Merged versions are edited directly and retain original record IDs, metadata,
source hashes/provenance, and legacy/pairwise sections. Duplicate IDs are addressed
by row within the snapshot. Skip applies to that row; another unskipped duplicate
can still be exported. The original partitions are unchanged. Legacy labels lacking
record metadata remain preserved but are not shown as editable version-2 records.
Concurrent external changes block saves until the file is reloaded.

## Export the reviewed version

```sh
uv run python annotation_tools/prepare_training_data.py \
  --data-dir data \
  --text-key title --output-dir data/training_data/review_TIMESTAMP
```

By default the exporter selects the newest timestamped merge in `data/merged_annotations/`
by the UTC creation timestamp in its filename, not its modification time. Editing an
older version does not promote it. The selected path is printed and recorded in the
report. If no timestamped version exists, the legacy `merged_annotations.json` is
used; if neither exists, original partitions are scanned. A broken newest version
causes an error rather than silently falling back.

Use `--annotation-file merged_annotations/<filename>.json` to pin a specific version.
Only the selected version is read, so original partitions cannot reintroduce its old
labels. It excludes skipped records before checking scores,
review status, or image availability, and reports `skipped_explicit`. Invalid skip
values are rejected. Existing deduplication, conflict exclusion, image-grouped splits,
and optional training-only augmentation remain available. Output directories must
be new. Configuration JSON and CLI overrides are supported.

## Tests

```sh
PYTHONPATH=src:annotation_tools:annotation_tools/annotator uv run pytest annotation_tools tests/test_merge_annotations.py -q
uv run ruff check annotation_tools
```
