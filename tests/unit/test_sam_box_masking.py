import numpy as np
from PIL import Image

from lerobot.utils.sam_box_masking import SamBoxMasker


def test_only_segmented_pixels_are_blacked_out_and_input_is_rgb():
    class Predictor:
        def set_image(self, rgb):
            np.testing.assert_array_equal(rgb[0, 0], [200, 100, 50])

        def predict(self, box, multimask_output):
            np.testing.assert_allclose(box, [[0, 0, 4, 4], [2, 2, 4, 4]])
            mask = np.zeros((2, 1, 4, 4), dtype=bool)
            mask[0, 0, 1, 1] = True
            mask[1, 0, 2, 2] = True
            return mask, None, None

    masker = SamBoxMasker.__new__(SamBoxMasker)
    masker.predictor = Predictor()
    image = Image.new('RGB', (4, 4), (200, 100, 50))
    boxes = np.array([[[0, 0, 1, 1], [.5, .5, 1, 1], [-1]*4]], dtype=np.float32)
    images, detected = masker.process([image], boxes)
    assert images[0].getpixel((1, 1)) == (0, 0, 0)
    assert images[0].getpixel((2, 2)) == (0, 0, 0)
    assert images[0].getpixel((0, 0)) == (200, 100, 50)
    assert image.getpixel((1, 1)) == (200, 100, 50)
    np.testing.assert_array_equal(detected, boxes)


def test_no_detection_preserves_image():
    masker = SamBoxMasker.__new__(SamBoxMasker)
    image = Image.new('RGB', (4, 4), (200, 100, 50))
    result, _ = masker.process([image], np.full((1, 4, 4), -1.0))
    np.testing.assert_array_equal(np.asarray(result[0]), np.asarray(image))
