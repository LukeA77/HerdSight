import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "make_lofo_sets", Path(__file__).parents[1] / "gpu" / "make_lofo_sets.py"
)
lofo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lofo)

DERVAL = "Derval__f__animal__0_0.jpg"
JALOGNY = "Jalogny__f__animal__0_0.jpg"
MAURON = "Mauron__f__empty__0_0.jpg"


def _write(pkg, k, names):
    images = [
        {"id": i + 1, "file_name": n, "width": 640, "height": 640} for i, n in enumerate(names)
    ]
    anns = [
        {"id": i + 1, "image_id": i + 1, "bbox": [1, 2, 3, 4], "category_id": 1}
        for i, n in enumerate(names)
        if "animal" in n
    ]
    c = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "cattle"}]}
    (pkg / f"train_fold{k}.json").write_text(json.dumps(c))


def _pkg(tmp_path):
    pkg = tmp_path / "pkg"
    (pkg / "images").mkdir(parents=True)
    (pkg / "manifest.json").write_text('{"package": "test"}')
    _write(pkg, 0, [JALOGNY, MAURON])
    _write(pkg, 1, [DERVAL, MAURON])
    _write(pkg, 2, [DERVAL, JALOGNY])
    return pkg


def test_held_out_farm_never_in_its_training_set(tmp_path):
    out = tmp_path / "out"
    try:
        counts = lofo.make_sets(_pkg(tmp_path), out)
    except OSError:
        pytest.skip("symbolic links not permitted here")
    assert counts["Derval"] == {"train_tiles": 2, "monitor_tiles": 1}
    assert (out / "manifest.json").read_text() == '{"package": "test"}'
    for farm in lofo.FARMS:
        train = json.loads((out / f"train_lofo_{farm}.json").read_text())
        names = [im["file_name"] for im in train["images"]]
        assert names and not any(n.startswith(farm + "__") for n in names)
        ids = {im["id"] for im in train["images"]}
        assert all(a["image_id"] in ids for a in train["annotations"])
    monitor = json.loads((out / "monitor_lofo_Mauron.json").read_text())
    assert monitor["images"] == []  # only tiles with animals


def test_unknown_farm_is_refused(tmp_path):
    pkg = _pkg(tmp_path)
    _write(pkg, 0, ["Other__f__animal__0_0.jpg"])
    with pytest.raises(SystemExit):
        lofo.make_sets(pkg, tmp_path / "out")
