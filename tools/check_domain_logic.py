"""无需权重的关键规则测试：无效区不评分、空预测不补真值、空标签筛除。"""

import tempfile, unittest
from pathlib import Path
import h5py, numpy as np
from xiaomi_lens.domain_evaluation import score
from xiaomi_lens.description_checks import consistency_issues, answer_presence
from xiaomi_lens.train.data import domain


class Rules(unittest.TestCase):
    def test_invalid_pixels_ignored(self):
        truth = np.array([[1, 0], [0, 0]], bool)
        valid = np.array([[1, 1], [0, 0]], bool)
        pred = np.array([[1, 0], [1, 1]], bool)
        self.assertEqual(score(pred, truth, valid)["iou"], 1.0)

    def test_empty_and_missed(self):
        z = np.zeros((2, 2), bool)
        v = np.ones((2, 2), bool)
        self.assertEqual(score(z, z, v)["iou"], 1.0)
        self.assertEqual(score(z, v, v)["iou"], 0.0)
        self.assertTrue(score(v, z, v)["false_positive_image"])

    def test_real_label_filter(self):
        original = domain.SOURCE
        with tempfile.TemporaryDirectory() as d:
            domain.SOURCE = Path(d)
            base = Path(d) / "synthetic"
            base.mkdir()
            with h5py.File(base / "mask.h5", "w") as f:
                f["mask"] = np.array([[1, 0], [0, 0]], np.uint8)
                f["valid_mask"] = np.array([[0, 1], [1, 1]], np.uint8)
            try:
                _, _, m, v = domain.read_source(
                    "synthetic", {"mask_hdf5": "mask.h5"}, False
                )
                self.assertFalse(m.any())
            finally:
                domain.SOURCE = original

    def test_tiny_and_disconnected_descriptions(self):
        m = np.zeros((1024, 1024), bool)
        m[0, 0] = True
        m[-1, -1] = True
        text = domain.describe(m)
        self.assertIn("小于0.1%", text)
        self.assertIn("多块分布", text)
        self.assertIn("整体中心", text)
        m[-1, -1] = False
        answer = domain.training_answer(m, np.ones(m.shape, bool))
        self.assertIn("<bbox>[0, 0, 1, 1]</bbox><|mask|>", answer)

    def test_description_area_is_not_prediction_quality(self):
        m = np.zeros((100, 100), bool)
        m[70:80, 70:80] = True
        self.assertTrue(consistency_issues("检测到滑坡区域，面积约10%。", m))
        self.assertFalse(
            consistency_issues(
                "检测到滑坡区域，区域整体中心位于图像右下部，呈连续分布，占有效图像面积约1.0%。",
                m,
            )
        )

    def test_answer_presence_independent_of_empty_mask(self):
        self.assertTrue(answer_presence("检测到滑坡区域。", 0))
        self.assertFalse(answer_presence("未检测到滑坡区域。", 0))
        self.assertIsNone(answer_presence("无法判断。", 0))

    def test_footprint_affine(self):
        box = domain.footprint({"geotransform": [0.01, 0, 100, 0, -0.01, 40]})
        self.assertAlmostEqual(box[0], 100)
        self.assertAlmostEqual(box[1], 38.72)


if __name__ == "__main__":
    unittest.main()
