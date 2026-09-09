import json
from pathlib import Path
import tempfile
import unittest
from server import Dataset


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root / "images").mkdir()
        (self.root / "images" / "a.png").write_bytes(b"synthetic")
        (self.root / "metadata.json").write_text(
            json.dumps(
                {
                    "items": [
                        {"id": "a", "image": "a.png", "text": "Synthetic A"},
                        {"id": "b", "text": "Synthetic B"},
                    ]
                }
            )
        )
        self.config = {"folder": str(self.root)}

    def test_save_reload_override_and_preserve_source(self):
        original = (self.root / "metadata.json").read_bytes()
        dataset = Dataset(self.config)
        self.assertEqual(dataset.paths["a"], self.root / "images" / "a.png")
        dataset.save({"mode": "pointwise", "id": "a", "score": 0, "reasoning": ""})
        dataset = Dataset(self.config)
        self.assertEqual(dataset.annotations["pointwise"]["a"]["score"], 0)
        dataset.save({"mode": "pointwise", "id": "a", "score": 9, "reasoning": "Updated"})
        dataset.save(
            {"mode": "pairwise", "left": "a", "right": "b", "choice": "tie", "reasoning": ""}
        )
        dataset = Dataset(self.config)
        self.assertEqual(
            dataset.annotations["pointwise"]["a"], {"score": 9, "reasoning": "Updated", "skip": False}
        )
        self.assertEqual(dataset.annotations["pairwise"]['["a","b"]']["choice"], "tie")
        self.assertEqual((self.root / "metadata.json").read_bytes(), original)

    def test_invalid_score_cannot_overwrite(self):
        dataset = Dataset(self.config)
        for score in (-1, 10, True, 1.5, None):
            with self.assertRaises(ValueError):
                dataset.save({"mode": "pointwise", "id": "a", "score": score})
        self.assertFalse(dataset.output.exists())

    def test_path_escape_and_duplicate_ids(self):
        with self.assertRaises(ValueError):
            Dataset({**self.config, "json_file": "../elsewhere.json"})
        (self.root / "metadata.json").write_text('[{"id":"a"},{"id":"a"}]')
        with self.assertRaises(ValueError):
            Dataset(self.config)

    def test_invalid_existing_output_preserved(self):
        (self.root / "annotations").mkdir()
        (self.root / "annotations/metadata.json").write_text("{broken")
        with self.assertRaises(ValueError):
            Dataset(self.config)
        self.assertEqual((self.root / "annotations/metadata.json").read_text(), "{broken")

    def test_custom_mapping(self):
        (self.root / "metadata.json").write_text(
            json.dumps({"nested": {"rows": [{"uid": 42, "media": {"path": "a.png"}}]}})
        )
        dataset = Dataset(
            {
                **self.config,
                "records_key": "nested.rows",
                "id_key": "uid",
                "image_key": "media.path",
            }
        )
        self.assertEqual(dataset.records[0]["id"], "42")
        self.assertTrue(dataset.paths["42"].is_file())


if __name__ == "__main__":
    unittest.main()


class PickerTests(unittest.TestCase):
    def test_home_relative_browse_and_image_url(self):
        from unittest.mock import patch
        from server import browse

        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory).resolve()
            dataset_root = home / "examples"
            (dataset_root / "images").mkdir(parents=True)
            (dataset_root / "images" / "sample.JPG").write_bytes(b"synthetic-jpg")
            (dataset_root / "images" / "sample.png").write_bytes(b"synthetic-png")
            (dataset_root / "meta.json").write_text(
                json.dumps(
                    [
                        {"id": "jpg", "image_url": "images/sample.JPG", "image": "wrong.png"},
                        {"id": "png", "image_url": "images/sample.png"},
                    ]
                )
            )
            with patch("server.Path.home", return_value=home):
                listing = browse("examples")
                self.assertEqual(listing["path"], "examples")
                self.assertEqual(listing["parent"], ".")
                self.assertEqual(listing["json_files"], ["meta.json"])
                self.assertEqual(listing["folders"], ["images"])
                dataset = Dataset({"folder": "examples", "json_file": "meta.json"})
                self.assertEqual(dataset.paths["jpg"], dataset_root / "images/sample.JPG")
                self.assertEqual(dataset.paths["png"], dataset_root / "images/sample.png")
                with self.assertRaises(ValueError):
                    browse("..")

    def test_listing_does_not_parse_json(self):
        from unittest.mock import patch
        from server import browse

        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory).resolve()
            (home / "broken.json").write_text("this is deliberately not JSON")
            (home / "annotations.json").write_text("also not JSON")
            with patch("server.Path.home", return_value=home):
                self.assertEqual(browse(".")["json_files"], ["broken.json"])


class IncrementalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root / "metadata").mkdir()
        (self.root / "images").mkdir()
        self.a = {"id": "shared", "image_url": "images/a.png", "title": "Synthetic"}
        (self.root / "images/a.png").write_bytes(b"synthetic")
        self.write("one", [self.a])

    def write(self, batch, records):
        (self.root / "metadata" / (batch + ".json")).write_text(json.dumps(records))

    def load(self, batch="one"):
        return Dataset({"folder": str(self.root), "json_file": batch + ".json"})

    def test_isolation_complete_export_append_reorder(self):
        one = self.load()
        one.save({"mode": "pointwise", "id": "shared", "score": 7, "reasoning": "test"})
        exported = json.loads((self.root / "annotations/one.json").read_text())
        self.assertEqual(exported["records"][0]["metadata"], self.a)
        self.assertEqual(exported["records"][0]["image_path"], "images/a.png")
        self.assertEqual(exported["records"][0]["annotation"]["score"], 7)
        self.write("two", [self.a])
        two = self.load("two")
        self.assertEqual(two.annotations["pointwise"], {})
        two.save({"mode": "pointwise", "id": "shared", "score": 2})
        self.assertEqual(self.load().annotations["pointwise"]["shared"]["score"], 7)
        self.write("one", [{"id": "new"}, self.a])
        self.assertEqual(self.load().annotations["pointwise"]["shared"]["score"], 7)
        self.assertEqual(len(json.loads(one.output.read_text())["records"]), 1)

    def test_fallback_ids_stable_after_reorder(self):
        self.write("one", [{"text": "a"}, {"text": "b"}])
        dataset = self.load()
        rid = dataset.records[0]["id"]
        dataset.save({"mode": "pointwise", "id": rid, "score": 0})
        self.write("one", [{"text": "b"}, {"text": "a"}, {"text": "c"}])
        restored = self.load()
        self.assertEqual(restored.records[1]["id"], rid)
        self.assertEqual(restored.annotations["pointwise"][rid]["score"], 0)

    def test_changed_metadata_and_external_save_blocked(self):
        first, second = self.load(), self.load()
        first.save({"mode": "pointwise", "id": "shared", "score": 8})
        with self.assertRaises(ValueError):
            second.save({"mode": "pointwise", "id": "shared", "score": 3})
        self.write("one", [{**self.a, "title": "changed"}])
        with self.assertRaises(ValueError):
            self.load()

    def test_metadata_folder_selection_and_legacy_preservation(self):
        (self.root / "annotations.json").write_text("legacy untouched")
        dataset = Dataset({"folder": str(self.root / "metadata"), "json_file": "one.json"})
        self.assertEqual(dataset.root, self.root)
        self.assertTrue(dataset.notice)
        dataset.save({"mode": "pointwise", "id": "shared", "score": 9})
        self.assertEqual((self.root / "annotations.json").read_text(), "legacy untouched")


class RelocatedLegacyTests(unittest.TestCase):
    def test_relocated_legacy_and_prefixed_metadata_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "metadata").mkdir()
            (root / "annotations").mkdir()
            (root / "metadata/batch.json").write_text('[{"id":"a"}]')
            legacy = root / "annotations/annotations.json"
            legacy.write_text("unread legacy fixture")
            dataset = Dataset(
                {"folder": str(root / "metadata"), "json_file": "metadata/batch.json"}
            )
            self.assertEqual(dataset.output, root / "annotations/batch.json")
            self.assertIn("annotations/annotations.json", dataset.notice)
            dataset.save({"mode": "pointwise", "id": "a", "score": 5})
            self.assertEqual(legacy.read_text(), "unread legacy fixture")


class LegacyImportTests(unittest.TestCase):
    def test_old_row_ids_map_to_new_ids_and_preserve_newer_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata").mkdir()
            (root / "metadata/batch.json").write_text(
                json.dumps(
                    [{"thread_id": "stable-a", "text": "synthetic A"}, {"text": "synthetic B"}]
                )
            )
            legacy = root / "annotations.json"
            legacy.write_text(
                json.dumps(
                    {
                        "pointwise": {
                            "0": {"score": 0, "reasoning": "old A"},
                            "1": {"score": 9, "reasoning": "old B"},
                        },
                        "pairwise": {},
                    }
                )
            )
            original = legacy.read_bytes()
            config = {"folder": str(root)}
            dataset = Dataset(config)
            self.assertEqual(dataset.import_legacy(legacy)["imported"], 2)
            reloaded = Dataset(config)
            self.assertEqual(
                reloaded.annotations["pointwise"]["stable-a"], {"score": 0, "reasoning": "old A", "skip": False}
            )
            self.assertEqual(len(reloaded.annotations["pointwise"]), 2)
            reloaded.save({"mode": "pointwise", "id": "stable-a", "score": 4, "reasoning": "new"})
            self.assertEqual(reloaded.import_legacy(legacy)["imported"], 0)
            self.assertEqual(Dataset(config).annotations["pointwise"]["stable-a"]["score"], 4)
            self.assertEqual(legacy.read_bytes(), original)
            legacy.write_text('{"pointwise":{"missing":{"score":3}}}')
            with self.assertRaises(ValueError):
                reloaded.import_legacy(legacy)


class SkipTests(unittest.TestCase):
    def test_skip_without_score_then_override(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "metadata.json").write_text(json.dumps([{"id": "a"}]))
            config = {"folder": folder}
            dataset = Dataset(config)
            dataset.save({"mode": "pointwise", "id": "a", "skip": True})
            loaded = Dataset(config)
            self.assertTrue(loaded.annotations["pointwise"]["a"]["skip"])
            with self.assertRaises(ValueError):
                loaded.save({"mode": "pointwise", "id": "a", "skip": False})
            with self.assertRaises(ValueError):
                loaded.save({"mode": "pointwise", "id": "a", "skip": "false", "score": 4})
            loaded.save({"mode": "pointwise", "id": "a", "skip": False,
                         "score": 4, "reasoning": "restored"})
            self.assertEqual(Dataset(config).annotations["pointwise"]["a"],
                             {"score": 4, "reasoning": "restored", "skip": False})

    def test_merged_duplicate_ids_preserve_envelope_and_detect_stale_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "merged_annotations").mkdir()
            path = root / "merged_annotations/merged_version.json"
            row = {"id": "duplicate", "metadata": {"title": "synthetic"},
                   "image_path": None, "annotation": {"score": 6, "reasoning": "keep"}}
            document = {"artifact_type": "merged_annotations", "schema_version": 2,
                        "records": [row, row], "legacy_partitions": [{"document": "keep"}],
                        "record_sources": [{"file": "a"}, {"file": "b"}]}
            path.write_text(json.dumps(document))
            config = {"folder": folder, "json_file": str(path.relative_to(root))}
            first, stale = Dataset(config), Dataset(config)
            self.assertEqual(len(first.records), 2)
            first.save({"mode": "pointwise", "id": "0", "score": 6,
                        "reasoning": "keep", "skip": True})
            saved = json.loads(path.read_text())
            self.assertTrue(saved["records"][0]["annotation"]["skip"])
            self.assertEqual(saved["records"][1], row)
            self.assertEqual(saved["legacy_partitions"], document["legacy_partitions"])
            self.assertEqual(saved["record_sources"], document["record_sources"])
            self.assertTrue(Dataset(config).annotations["pointwise"]["0"]["skip"])
            with self.assertRaises(ValueError):
                stale.save({"mode": "pointwise", "id": "1", "score": 5})
            with self.assertRaises(ValueError):
                first.save({"mode": "pairwise", "left": "0", "right": "1", "choice": "tie"})
