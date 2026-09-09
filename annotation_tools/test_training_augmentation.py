import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from prepare_training_data import prepare
from training_augmentation import validate_augmentation


class AugmentationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "annotations").mkdir()
        (self.root / "images").mkdir()
        records = []
        for i in range(10):
            im = Image.new("RGB", (8, 6))
            im.putdata([(20 + i * 3 + x * 5, 70 + y * 5, 100) for y in range(6) for x in range(8)])
            im.save(self.root / f"images/{i}.png")
            records.append(
                {
                    "id": str(i),
                    "metadata": {"text": "hello   world"},
                    "image_path": f"images/{i}.png",
                    "annotation": {"score": 7, "reasoning": "synthetic explanation"},
                }
            )
        (self.root / "annotations/batch.json").write_text(
            json.dumps({"schema_version": 2, "records": records})
        )
        self.config = {
            "data_dir": str(self.root),
            "output_dir": str(self.root / "baseline"),
            "ratios": {"train": 0.6, "validation": 0.2, "test": 0.2},
        }

    def rows(self, folder, split):
        return [
            json.loads(line)
            for line in (self.root / folder / (split + ".jsonl")).read_text().splitlines()
        ]

    def test_train_only_provenance_originals_and_seed(self):
        before = {p.name: p.read_bytes() for p in (self.root / "images").iterdir()}
        prepare(self.config)
        settings = {"enabled": True, "copies_per_sample": 2, "brightness": [0.95, 1.05]}
        config = {**self.config, "output_dir": str(self.root / "aug"), "augmentation": settings}
        report = prepare(config)
        for split in ("validation", "test", "all"):
            self.assertEqual(self.rows("baseline", split), self.rows("aug", split))
        original = self.rows("aug", "train_original")
        variants = [r for r in self.rows("aug", "train") if "parent_id" in r]
        self.assertGreater(len(variants), 0)
        self.assertEqual(report["exported_counts"]["train"], len(original) + len(variants))
        self.assertTrue(all(r["parent_id"] in {o["id"] for o in original} for r in variants))
        for row in variants:
            self.assertNotIn("reasoning", row)
            self.assertEqual(row["score"], 7)
            self.assertTrue((self.root / "aug" / row["image"]).is_file())
        self.assertEqual(before, {p.name: p.read_bytes() for p in (self.root / "images").iterdir()})
        prepare({**config, "output_dir": str(self.root / "again")})
        self.assertEqual(self.rows("aug", "train"), self.rows("again", "train"))

    def test_text_only_duplicates_skipped(self):
        prepare(
            {
                **self.config,
                "output_dir": str(self.root / "text"),
                "augmentation": {
                    "enabled": True,
                    "normalize_whitespace": True,
                    "copies_per_sample": 3,
                },
            }
        )
        variants = [r for r in self.rows("text", "train") if "parent_id" in r]
        self.assertEqual(len(variants), 6)
        self.assertTrue(all(r["text"] == "hello world" for r in variants))
        self.assertFalse((self.root / "text/augmented_images").exists())

    def test_flip_pixels(self):
        prepare(
            {
                **self.config,
                "output_dir": str(self.root / "flip"),
                "augmentation": {"enabled": True, "horizontal_flip": True},
            }
        )
        rows = self.rows("flip", "train")
        originals = {r["id"]: r for r in rows if "parent_id" not in r}
        for row in rows:
            if "parent_id" not in row:
                continue
            with Image.open(self.root / "flip" / row["image"]) as flipped:
                with Image.open(
                    self.root / "flip" / originals[row["parent_id"]]["image"]
                ) as original:
                    self.assertEqual(flipped.getpixel((0, 0)), original.getpixel((7, 0)))

    def test_generated_evaluation_collision_is_skipped(self):
        from training_augmentation import augment_train
        from prepare_training_data import digest_file

        prepare(self.config)
        parent = self.rows("baseline", "train")[0]
        with Image.open((self.root / "baseline" / parent["image"]).resolve()) as im:
            im.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(self.root / "reserved.png")
        staging = self.root / "stage"
        staging.mkdir()
        variants, stats = augment_train(
            [parent],
            self.root / "baseline",
            staging,
            {"enabled": True, "horizontal_flip": True},
            42,
            {digest_file(self.root / "reserved.png")},
        )
        self.assertEqual(variants, [])
        self.assertEqual(stats["skipped_reserved"], 1)

    def test_invalid_and_corrupt_are_transactional(self):
        for config in (
            {"enabled": True},
            {"enabled": True, "brightness": [0, 5]},
            {"enabled": True, "horizontal_flip": "yes"},
            {"unknown": True},
        ):
            with self.assertRaises(ValueError):
                validate_augmentation(config)
        for p in (self.root / "images").iterdir():
            p.write_bytes(b"not an image")
        output = self.root / "bad"
        with self.assertRaises(ValueError):
            prepare(
                {
                    **self.config,
                    "output_dir": str(output),
                    "augmentation": {"enabled": True, "horizontal_flip": True},
                }
            )
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
