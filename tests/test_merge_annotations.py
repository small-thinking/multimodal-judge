import json

import pytest

from multimodal_judge.merge_annotations import merge_annotations


def test_lossless_repeatable_merge(tmp_path):
    source = tmp_path / "annotations"
    source.mkdir()
    row = {"id": "same", "annotation": {"score": 5, "review_required": True}}
    partition = {"schema_version": 2, "records": [row], "pairwise": {"a": "b"}}
    for name in ("a.json", "b.json"):
        (source / name).write_text(json.dumps(partition))
    legacy = {"pointwise": {"0": {"score": 1}}, "pairwise": {}}
    (source / "legacy.json").write_text(json.dumps(legacy))
    (source / "a.json.bak").write_text("invalid backup")
    (source / "merged_old.json").write_text("ignored")
    output = tmp_path / "merged_annotations.json"
    result = merge_annotations(source, output)
    first = output.read_bytes()
    (source / "renamed.json").write_bytes(first)
    assert merge_annotations(source, output) == result
    assert output.read_bytes() == first
    assert result["records"] == [row, row]
    assert result["record_sources"] == [
        {"file": "a.json", "record_index": 0}, {"file": "b.json", "record_index": 0}]
    assert result["legacy_partitions"][0]["document"] == legacy
    assert result["pairwise_partitions"][0]["annotations"] == {"a": "b"}


def test_invalid_input_preserves_output(tmp_path):
    source = tmp_path / "annotations"
    source.mkdir()
    output = tmp_path / "merged_annotations.json"
    output.write_text("previous result")
    with pytest.raises(ValueError, match="No annotation"):
        merge_annotations(source, output)
    (source / "broken.json").write_text('{"schema_version": 2}')
    with pytest.raises(ValueError, match="records"):
        merge_annotations(source, output)
    assert output.read_text() == "previous result"
    with pytest.raises(ValueError, match="outside"):
        merge_annotations(source, source / "merged_annotations.json")
    with pytest.raises(ValueError, match="named"):
        merge_annotations(source, tmp_path / "ordinary.json")


def test_cli_default_creates_distinct_versions(tmp_path, monkeypatch):
    from multimodal_judge.merge_annotations import main

    source = tmp_path / "annotations"
    source.mkdir()
    (source / "a.json").write_text(json.dumps({"schema_version": 2, "records": []}))
    monkeypatch.setattr("sys.argv", ["merge_annotations", "--annotation-dir", str(source)])
    main()
    main()
    outputs = list((tmp_path / "merged_annotations").glob("merged_annotations_*.json"))
    assert len(outputs) == 2
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
