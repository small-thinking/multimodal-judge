"""Synthetic-only regression tests for the local training-data exporter.

Run: python -m unittest discover -s tools -p 'test_prepare_training_data.py'
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE_PATH = Path(__file__).with_name("prepare_training_data.py")
spec = importlib.util.spec_from_file_location("prepare_training_data", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
prepare = module.prepare


class PrepareTrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        (self.data / "images").mkdir(parents=True)
        (self.data / "annotations").mkdir()
        self.output = self.root / "export"

    def image(self, name, content=b"synthetic image bytes"):
        path = self.data / "images" / name
        path.write_bytes(content)
        return "images/" + name

    def record(self, identity, image, text="A synthetic prompt", score=7, reasoning=""):
        return {"id": identity, "metadata": {"text": text}, "image_path": image,
                "annotation": {"score": score, "reasoning": reasoning}}

    def batch(self, name, records):
        path = self.data / "annotations" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema_version": 2, "records": records}), encoding="utf-8")

    def config(self, **overrides):
        config = {"data_dir": str(self.data), "output_dir": str(self.output),
                  "ratios": {"train": .8, "validation": .1, "test": .1}, "seed": 42,
                  "text_key": "text", "allow_empty_text": False, "include_reasoning": True}
        config.update(overrides)
        return config

    def rows(self, name, output=None):
        return [json.loads(line) for line in ((output or self.output) / name).read_text().splitlines() if line]

    def test_unscored_skipped_but_zero_kept(self):
        image = self.image("valid.png")
        missing_annotation = {"id": "missing"}
        missing_score = {"id": "empty", "annotation": {"reasoning": "only reasoning"}}
        null_annotation = {"id": "null-annotation", "annotation": None}
        records = [missing_annotation, missing_score, null_annotation]
        records += [self.record(str(i), "missing.png", score=score) for i, score in enumerate([None, "", "  "])]
        records += [self.record("zero", image, score=0), self.record("bad", image, score=False)]
        self.batch("batch.json", records)
        report = prepare(self.config())
        self.assertEqual(report["skipped_unscored"], 6)
        self.assertEqual(report["rejected_records"], 1)
        self.assertEqual([r["score"] for r in self.rows("all.jsonl")], [0])
        self.assertEqual(len(self.rows("conflicts.jsonl")), 0)

    def test_explicit_skip_precedes_other_validation_and_can_be_cleared(self):
        record = self.record("skip", self.image("valid.png"))
        record["annotation"].update(skip=True, review_required=True, score="bad")
        record["image_path"] = "missing.png"
        self.batch("one.json", [record])
        report = prepare(self.config())
        self.assertEqual(report["skipped_explicit"], 1)
        self.assertEqual(report["skipped_unreviewed"], 0)
        self.assertEqual(report["rejected_records"], 0)
        self.assertEqual(self.rows("all.jsonl"), [])
        record["annotation"].update(skip=False, review_required=False, score=0)
        record["image_path"] = "images/valid.png"
        self.batch("one.json", [record])
        other = self.root / "unskipped"
        report = prepare(self.config(output_dir=str(other)))
        self.assertEqual(report["skipped_explicit"], 0)
        self.assertEqual(self.rows("all.jsonl", other)[0]["score"], 0)

    def test_invalid_skip_types_are_rejected(self):
        records = []
        for index, value in enumerate([None, 0, 1, "false", [], {}]):
            record = {"id": str(index), "annotation": {"skip": value}}
            records.append(record)
        self.batch("one.json", records)
        report = prepare(self.config())
        self.assertEqual(report["rejected_records"], 6)
        self.assertEqual({r["reason"] for r in self.rows("rejected.jsonl")}, {"invalid_skip"})

    def test_selected_merged_input_ignores_original_partitions(self):
        image = self.image("valid.png")
        self.batch("original.json", [self.record("original", image, score=2)])
        self.batch("merged_review/selected.json", [self.record("merged", image, score=9)])
        selected = self.data / "annotations/merged_review/selected.json"
        document = json.loads(selected.read_text())
        document["artifact_type"] = "merged_annotations"
        document["record_sources"] = [{"file": "original.json", "record_index": 0}]
        selected.write_text(json.dumps(document))
        report = prepare(self.config(annotation_file="annotations/merged_review/selected.json"))
        self.assertEqual(report["input_records"], 1)
        self.assertEqual(self.rows("all.jsonl")[0]["sources"][0]["original_source"],
                         {"file": "original.json", "record_index": 0})
        self.assertEqual([r["score"] for r in self.rows("all.jsonl")], [9])
        self.assertEqual([r["path"] for r in report["input_files"]], ["annotations/merged_review/selected.json"])

    def test_default_scan_excludes_merged_paths_and_artifact_type(self):
        image = self.image("valid.png")
        self.batch("original.json", [self.record("original", image, score=2)])
        self.batch("merged_review/nested.json", [self.record("nested", image, score=8)])
        self.batch("merged_output.json", [self.record("named", image, score=8)])
        self.batch("renamed.json", [self.record("artifact", image, score=8)])
        artifact = self.data / "annotations/renamed.json"
        document = json.loads(artifact.read_text())
        document["artifact_type"] = "merged_annotations"
        artifact.write_text(json.dumps(document))
        report = prepare(self.config())
        self.assertEqual(report["input_records"], 1)
        self.assertEqual(len(report["skipped_files"]), 3)
        self.assertEqual([r["score"] for r in self.rows("all.jsonl")], [2])

    def test_selected_root_file_needs_no_partitions_directory(self):
        (self.data / "annotations").rmdir()
        selected = self.data / "selected.json"
        selected.write_text(json.dumps({"schema_version": 2, "records": [
            self.record("selected", self.image("valid.png"))]}))
        output = self.data / "training_export"
        report = prepare(self.config(annotation_file="selected.json", output_dir=str(output)))
        self.assertEqual(report["unique_samples"], 1)
        self.assertEqual(len(self.rows("all.jsonl", output)), 1)

    def test_selected_input_path_and_output_boundaries(self):
        outside = self.root / "outside.json"
        outside.write_text('{"schema_version": 2, "records": []}')
        with self.assertRaises(ValueError):
            prepare(self.config(annotation_file=str(outside)))
        self.batch("one.json", [])
        selected = self.data / "annotations/one.json"
        with self.assertRaises(FileExistsError):
            prepare(self.config(annotation_file=str(selected), output_dir=str(selected)))
        with self.assertRaises(ValueError):
            prepare(self.config(annotation_file=str(selected), output_dir=str(selected.parent / "export")))

    def test_ai_draft_is_excluded_until_reviewed(self):
        image = self.image("draft.png")
        draft = self.record("draft", image)
        draft["annotation"]["review_required"] = True
        self.batch("batch.json", [draft])
        report = prepare(self.config())
        self.assertEqual(report["skipped_unreviewed"], 1)
        self.assertEqual(self.rows("all.jsonl"), [])

    def test_same_bytes_across_filenames_and_batches_merge(self):
        first = self.image("first.png")
        second = self.image("second.jpg")
        self.batch("one.json", [self.record("first", first, " prompt ", reasoning="reason one")])
        self.batch("nested/two.json", [self.record("second", second, "prompt", reasoning="reason two")])
        prepare(self.config())
        rows = self.rows("all.jsonl")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["text"], "prompt")
        self.assertEqual(rows[0]["score"], 7)
        self.assertEqual(len(rows[0]["sources"]), 2)
        self.assertEqual((self.output / rows[0]["image"]).resolve().read_bytes(), b"synthetic image bytes")
        self.assertIn("reasoning", rows[0])
        self.assertTrue((self.output / "report.json").exists())

    def test_same_image_different_text_cannot_cross_splits(self):
        records = []
        for index in range(20):
            path = self.image(f"{index}.png", f"image {index}".encode())
            records.extend([self.record(f"{index}-a", path, "first prompt"),
                            self.record(f"{index}-b", path, "second prompt")])
        self.batch("batch.json", records)
        prepare(self.config())
        images_to_split = {}
        all_ids = []
        for split in ("train", "validation", "test"):
            for row in self.rows(split + ".jsonl"):
                image_bytes = (self.output / row["image"]).resolve().read_bytes()
                if image_bytes in images_to_split:
                    self.assertEqual(images_to_split[image_bytes], split)
                images_to_split[image_bytes] = split
                all_ids.append(row["id"])
        self.assertEqual(len(all_ids), 40)
        self.assertEqual(len(set(all_ids)), 40)
        self.assertEqual(len(images_to_split), 20)

    def test_score_conflicts_excluded(self):
        path = self.image("conflict.png")
        self.batch("one.json", [self.record("a", path, score=2)])
        self.batch("two.json", [self.record("b", path, score=8)])
        prepare(self.config())
        self.assertEqual(self.rows("all.jsonl"), [])
        self.assertTrue(self.rows("conflicts.jsonl"))
        for split in ("train", "validation", "test"):
            self.assertEqual(self.rows(split + ".jsonl"), [])

    def test_invalid_records_rejected(self):
        path = self.image("valid.png")
        self.batch("one.json", [self.record("valid", path),
                                self.record("missing-image", "images/absent.png"),
                                self.record("empty-text", path, "  "),
                                self.record("invalid-score", path, "other", score=12)])
        prepare(self.config())
        self.assertEqual(len(self.rows("all.jsonl")), 1)
        self.assertEqual(len(self.rows("rejected.jsonl")), 3)

    def test_same_seed_produces_identical_split_membership(self):
        self.batch("one.json", [self.record(str(i), self.image(f"{i}.jpg", str(i).encode())) for i in range(30)])
        prepare(self.config())
        other = self.root / "other-export"
        prepare(self.config(output_dir=str(other)))
        for split in ("train", "validation", "test"):
            self.assertEqual([r["id"] for r in self.rows(split + ".jsonl")],
                             [r["id"] for r in self.rows(split + ".jsonl", other)])

    def test_configurable_ratios_allow_zero_test(self):
        self.batch("one.json", [self.record(str(i), self.image(f"{i}.png", str(i).encode())) for i in range(10)])
        prepare(self.config(ratios={"train": .6, "validation": .4, "test": 0}))
        self.assertEqual(len(self.rows("train.jsonl")), 6)
        self.assertEqual(len(self.rows("validation.jsonl")), 4)
        self.assertEqual(self.rows("test.jsonl"), [])

    def test_invalid_ratios_are_rejected(self):
        self.batch("one.json", [self.record("a", self.image("one.png"))])
        for ratios in ({"train": .9, "validation": .2, "test": .1},
                       {"train": -.1, "validation": .9, "test": .2},
                       {"train": float("nan"), "validation": 0, "test": 0}):
            with self.subTest(ratios=ratios), self.assertRaises(ValueError):
                prepare(self.config(ratios=ratios))
        self.assertFalse((self.output / "all.jsonl").exists())

    def test_existing_artifacts_are_never_overwritten(self):
        self.batch("one.json", [self.record("a", self.image("one.png"))])
        prepare(self.config())
        before = {p.name: p.read_bytes() for p in self.output.iterdir() if p.is_file()}
        with self.assertRaises(FileExistsError):
            prepare(self.config())
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.output.iterdir() if p.is_file()})

    def test_legacy_file_is_skipped_without_reimporting_labels(self):
        self.batch("one.json", [self.record("a", self.image("one.png"))])
        legacy = self.data / "annotations" / "legacy.json"
        legacy.write_text(json.dumps({"pointwise": {"0": {"score": 3, "reasoning": "old"}}}))
        prepare(self.config())
        self.assertEqual(len(self.rows("all.jsonl")), 1)
        self.assertIn("legacy.json", (self.output / "report.json").read_text())

    def test_custom_text_field_and_optional_reasoning(self):
        record = self.record("a", self.image("one.png"))
        record["metadata"] = {"caption": "Custom text"}
        self.batch("one.json", [record])
        prepare(self.config(text_key="caption", include_reasoning=False))
        row = self.rows("all.jsonl")[0]
        self.assertEqual(row["text"], "Custom text")
        self.assertNotIn("reasoning", row)


if __name__ == "__main__":
    unittest.main()
