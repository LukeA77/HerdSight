import numpy as np
from PIL import Image

from herdsight.detect.run_tiled import detect_full_image


def test_animal_seen_by_several_tiles_is_reported_once():
    animals = np.array([[530, 100, 559, 130], [561, 100, 590, 130], [900, 400, 930, 430]], float)

    def oracle(tile_img):
        # The "detector" returns every animal wholly inside the tile it is given, in tile pixels.
        x0, y0 = tile_img.info["origin"]
        w, h = tile_img.size
        inside = (
            (animals[:, 0] >= x0)
            & (animals[:, 2] <= x0 + w)
            & (animals[:, 1] >= y0)
            & (animals[:, 3] <= y0 + h)
        )
        return animals[inside] - [x0, y0, x0, y0], np.ones(inside.sum())

    image = Image.new("RGB", (1120, 640))
    original_crop = image.crop

    def crop(box):
        c = original_crop(box)
        c.info["origin"] = box[:2]
        return c

    image.crop = crop
    boxes, scores = detect_full_image(image, oracle, tile=640, overlap=160)
    order = np.argsort(boxes[:, 0])
    np.testing.assert_allclose(boxes[order], animals)
    assert scores.shape == (3,)
