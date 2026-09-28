import unittest
import tempfile

from PIL import Image
from pathlib import Path

from outputs.solution import (
    group_annotations_by_image,
    letterbox,
    p180_from_logits,
    resolve_textocr_image,
    validate_predictions,
)


class OrientationHelpersTests(unittest.TestCase):
    def test_textocr_image_resolver_supports_official_archive_folder_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = root / "train_images" / "example.jpg"
            expected.parent.mkdir()
            expected.touch()

            resolved = resolve_textocr_image(root, Path("train/example.jpg"))

            self.assertEqual(resolved, expected)

    def test_cache_order_keeps_annotations_for_same_image_adjacent(self):
        items = [
            (Path("b.jpg"), (0.0, 0.0, 20.0, 10.0)),
            (Path("a.jpg"), (0.0, 0.0, 20.0, 10.0)),
            (Path("b.jpg"), (20.0, 0.0, 20.0, 10.0)),
        ]

        grouped = group_annotations_by_image(items)

        self.assertEqual([item[0].name for item in grouped], ["a.jpg", "b.jpg", "b.jpg"])

    def test_letterbox_returns_fixed_grayscale_shape_without_stretching(self):
        image = Image.new("RGB", (120, 30), (255, 255, 255))
        pixels = letterbox(image, width=224, height=64)

        self.assertEqual(pixels.shape, (64, 224))
        self.assertEqual(pixels.dtype.name, "uint8")
        self.assertGreater(pixels[32].mean(), pixels[0].mean())

    def test_rotation_pair_probabilities_are_complementary(self):
        p = p180_from_logits(logit_input=1.2, logit_rot180=-0.4, temperature=0.8)
        p_after_rotation = p180_from_logits(
            logit_input=-0.4, logit_rot180=1.2, temperature=0.8
        )

        self.assertAlmostEqual(p + p_after_rotation, 1.0, places=12)
        self.assertGreater(p, 0.5)

    def test_predictions_reject_nan_and_out_of_range_values(self):
        for values in ([0.1, float("nan")], [-0.1, 0.5], [0.2, 1.1]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_predictions(["a", "b"], values)

    def test_predictions_accept_finite_probabilities_in_order(self):
        ids, probabilities = validate_predictions(["b", "a"], [0.0, 1.0])

        self.assertEqual(ids, ["b", "a"])
        self.assertEqual(probabilities, [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
