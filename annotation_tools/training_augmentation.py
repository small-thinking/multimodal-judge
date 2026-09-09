"""Deterministic, train-only augmentation for local training-data builds.

This module never alters source files. Generated images are written into the
caller's staging directory, so failed builds can be discarded as a unit.
"""

import copy
import hashlib
import json
import math
from pathlib import Path
import random


def validate_augmentation(config):
    defaults = {
        "enabled": False,
        "copies_per_sample": 1,
        "brightness": [1.0, 1.0],
        "contrast": [1.0, 1.0],
        "horizontal_flip": False,
        "normalize_whitespace": False,
        "preserve_reasoning": False,
    }
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError("augmentation must be an object.")
    if set(config) - set(defaults):
        raise ValueError("Unknown augmentation configuration key.")
    settings = {**defaults, **config}
    for key in ("enabled", "horizontal_flip", "normalize_whitespace", "preserve_reasoning"):
        if type(settings[key]) is not bool:
            raise ValueError(f"augmentation.{key} must be boolean.")
    count = settings["copies_per_sample"]
    if type(count) is not int or not 1 <= count <= 5:
        raise ValueError("augmentation.copies_per_sample must be an integer from 1 to 5.")
    for key in ("brightness", "contrast"):
        bounds = settings[key]
        if (
            not isinstance(bounds, (list, tuple))
            or len(bounds) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds)
            or not 0.8 <= bounds[0] <= bounds[1] <= 1.2
        ):
            raise ValueError(f"augmentation.{key} must be [low, high] within 0.8 to 1.2.")
        settings[key] = [float(v) for v in bounds]
    if settings["enabled"] and not (
        settings["horizontal_flip"]
        or settings["normalize_whitespace"]
        or settings["brightness"] != [1.0, 1.0]
        or settings["contrast"] != [1.0, 1.0]
    ):
        raise ValueError("Enabled augmentation requires at least one nonidentity operation.")
    return settings


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def augment_train(rows, output: Path, staging: Path, settings, seed, reserved_hashes: set):
    """Return additional rows and stats; never include validation/test originals.

    ``output`` is the final dataset path (source image references are relative to
    it); ``staging`` holds the not-yet-published build. ``reserved_hashes`` must
    include every validation/test image digest.
    """
    settings = validate_augmentation(settings)
    stats = {
        "enabled": settings["enabled"],
        "attempted": 0,
        "generated": 0,
        "skipped_noop": 0,
        "skipped_duplicate": 0,
        "skipped_reserved": 0,
        "pillow_version": None,
    }
    if not settings["enabled"]:
        return [], stats
    image_operations = (
        settings["horizontal_flip"]
        or settings["brightness"] != [1.0, 1.0]
        or settings["contrast"] != [1.0, 1.0]
    )
    if image_operations:
        try:
            import PIL
            from PIL import Image, ImageEnhance, ImageOps
        except ImportError:
            raise ValueError(
                "Image augmentation requires Pillow. Install it in this environment "
                "(uv pip install Pillow), then retry."
            ) from None
        stats["pillow_version"] = PIL.__version__
    variants = []
    variant_ids = set()
    seen = {(row["image_sha256"], row["text"]) for row in rows}
    for parent in rows:
        original = None
        if image_operations:
            try:
                with Image.open((output / parent["image"]).resolve()) as loaded:
                    if getattr(loaded, "n_frames", 1) != 1:
                        raise ValueError("animated")
                    oriented = ImageOps.exif_transpose(loaded)
                    original = oriented.convert(
                        "RGBA"
                        if "A" in oriented.getbands() or "transparency" in oriented.info
                        else "RGB"
                    )
                    original.load()
            except Exception:
                raise ValueError(
                    "Cannot augment a source image: decoding failed or image is animated. "
                    "Check source images locally; no build was published."
                ) from None
        for number in range(settings["copies_per_sample"]):
            stats["attempted"] += 1
            rng = random.Random(_digest([seed, parent["id"], number]))
            operations = {}
            text = parent["text"]
            if settings["normalize_whitespace"]:
                normalized = " ".join(text.split())
                if normalized != text:
                    operations["normalize_whitespace"] = True
                text = normalized
            transformed = original.copy() if original is not None else None
            if image_operations:
                for operation, enhancer in (
                    ("brightness", ImageEnhance.Brightness),
                    ("contrast", ImageEnhance.Contrast),
                ):
                    low, high = settings[operation]
                    factor = rng.uniform(low, high)
                    if factor != 1.0:
                        transformed = enhancer(transformed).enhance(factor)
                        operations[operation] = factor
                if settings["horizontal_flip"]:
                    transformed = ImageOps.mirror(transformed)
                    operations["horizontal_flip"] = True
            pixels_changed = original is not None and transformed.tobytes() != original.tobytes()
            if not pixels_changed and text == parent["text"]:
                stats["skipped_noop"] += 1
                continue
            variant_id = _digest([parent["id"], operations])
            image_hash = parent["image_sha256"]
            image_path = parent["image"]
            generated_path = None
            if pixels_changed:
                image_path = f"augmented_images/{variant_id}.png"
                generated_path = staging / image_path
                generated_path.parent.mkdir(parents=True, exist_ok=True)
                # Repeated deterministic operations can produce the same variant.
                # Check before writing to avoid deleting an earlier accepted image.
                if variant_id in variant_ids:
                    stats["skipped_duplicate"] += 1
                    continue
                transformed.save(generated_path, format="PNG")
                image_hash = hashlib.sha256(generated_path.read_bytes()).hexdigest()
            if image_hash in reserved_hashes:
                stats["skipped_reserved"] += 1
                if generated_path is not None:
                    generated_path.unlink()
                continue
            key = (image_hash, text)
            if key in seen:
                stats["skipped_duplicate"] += 1
                if generated_path is not None:
                    generated_path.unlink()
                continue
            seen.add(key)
            variant = copy.deepcopy(parent)
            if not settings["preserve_reasoning"]:
                variant.pop("reasoning", None)
                variant.pop("reasoning_alternatives", None)
            variant.update(
                id=variant_id,
                image=image_path,
                image_sha256=image_hash,
                text=text,
                parent_id=parent["id"],
                parent_image_sha256=parent["image_sha256"],
                augmentation=operations,
            )
            variants.append(variant)
            variant_ids.add(variant_id)
            stats["generated"] += 1
    return variants, stats
