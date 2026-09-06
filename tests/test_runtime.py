from types import SimpleNamespace

import pytest

from multimodal_judge.cli import select_device


def backend(cuda=False, mps=False):
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda),
                           backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)))


@pytest.mark.parametrize("cuda,mps,expected", [(True, True, "cuda"), (False, True, "mps"),
                                              (False, False, "cpu")])
def test_auto_priority(cuda, mps, expected):
    assert select_device("auto", backend(cuda, mps)) == expected


def test_explicit_gpu_never_silently_falls_back():
    with pytest.raises(ValueError, match="unavailable"):
        select_device("cuda", backend())
