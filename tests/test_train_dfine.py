"""Training-set construction in gpu/train_dfine.py (no GPU, no D-FINE needed)."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train_dfine", REPO / "gpu" / "train_dfine.py")
train_dfine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_dfine)

# Jalogny/20230627 is in fold1 and Jalogny/20230623 in fold0 (results/splits.json)
FOLD1_TILE = "Jalogny__DJI_202306271323_032__DJI_20230627132910_0042_V__0_0.jpg"
FOLD0_TILE = "Jalogny__DJI_202306231103_031__DJI_20230623111019_0042_V__640_0.jpg"


def _package(tmp: Path) -> Path:
    pkg = tmp / "pkg"
    (pkg / "images").mkdir(parents=True)
    coco = {
        "images": [{"id": 1, "file_name": FOLD1_TILE, "width": 640, "height": 640}],
        "annotations": [
            {"id": 1, "image_id": 1, "bbox": [10, 10, 50, 30], "category_id": 0, "area": 1500}
        ],
        "categories": [{"id": 0, "name": "cattle"}],
    }
    (pkg / "train_fold2.json").write_text(json.dumps(coco))
    return pkg


def _hardneg(tmp: Path, tiles: list[str]) -> Path:
    hn = tmp / "hn"
    (hn / "images").mkdir(parents=True)
    images = [
        {"id": i, "file_name": t, "width": 640, "height": 640} for i, t in enumerate(tiles, 1)
    ]
    (hn / "hardneg_fold2.json").write_text(json.dumps({"images": images, "annotations": []}))
    return hn


def _args(pkg: Path, hn: Path | None, exclude: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        package=pkg,
        fold="fold2",
        exclude=exclude,
        only_fold=None,
        holdout_group=None,
        oversample_touching=1,
        hardneg=hn,
    )


def test_hard_negatives_added_as_background(tmp_path: Path) -> None:
    pkg = _package(tmp_path)
    hn = _hardneg(tmp_path, [FOLD0_TILE, FOLD1_TILE])
    train_path, _, counts = train_dfine.build_sets(_args(pkg, hn), tmp_path)
    train = json.loads(train_path.read_text())
    assert counts["hard_negative_tiles"] == 2
    assert len(train["images"]) == 3
    assert len(train["annotations"]) == 1  # hard negatives carry no boxes
    names = [im["file_name"] for im in train["images"]]
    assert names[1:] == [f"hardneg/{FOLD0_TILE}", f"hardneg/{FOLD1_TILE}"]
    assert len({im["id"] for im in train["images"]}) == 3


def test_no_hard_negatives_by_default(tmp_path: Path) -> None:
    train_path, _, counts = train_dfine.build_sets(_args(_package(tmp_path), None), tmp_path)
    assert counts["hard_negative_tiles"] == 0
    assert len(json.loads(train_path.read_text())["images"]) == 1


def test_hard_negative_from_own_fold_refused(tmp_path: Path) -> None:
    # Jalogny/20230915 is in fold2, the fold being trained for
    own = "Jalogny__DJI_202309151045_055__DJI_20230926114121_0003_V__0_0.jpg"
    hn = _hardneg(tmp_path, [own])
    with pytest.raises(SystemExit, match="own images"):
        train_dfine.build_sets(_args(_package(tmp_path), hn), tmp_path)


def test_hard_negative_from_excluded_photo_refused(tmp_path: Path) -> None:
    exclude = tmp_path / "exclude.json"
    exclude.write_text(json.dumps(["Jalogny/DJI_202306231103_031/DJI_20230623111019_0042_V.JPG"]))
    hn = _hardneg(tmp_path, [FOLD0_TILE])
    with pytest.raises(SystemExit, match="excluded photo"):
        train_dfine.build_sets(_args(_package(tmp_path), hn, exclude), tmp_path)


def test_link_hardneg_makes_images_visible(tmp_path: Path) -> None:
    pkg = _package(tmp_path)
    hn = _hardneg(tmp_path, [FOLD0_TILE])
    (hn / "images" / FOLD0_TILE).write_bytes(b"jpg")
    train_dfine.link_hardneg(pkg, hn)
    assert (pkg / "images" / "hardneg" / FOLD0_TILE).read_bytes() == b"jpg"
