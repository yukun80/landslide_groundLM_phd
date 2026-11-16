"""停止规则真实场景与保存恢复检查，不需要加载大模型。"""
import json
import unittest
from xiaomi_lens.train.stopping import StopState


class StopTests(unittest.TestCase):
    def run_scores(self, scores):
        state = StopState()
        flags = [state.update(i, s) for i, s in enumerate(scores)]
        return state, flags

    def test_improvement(self):
        s, flags = self.run_scores([i * .01 for i in range(20)])
        self.assertFalse(any(flags))
        self.assertEqual(s.outcome(False), "epoch_limit_not_confirmed_stable")

    def test_plateau_and_minimum(self):
        s, flags = self.run_scores([.1, .2, .2002, .2004, .2006, .2008])
        self.assertEqual(flags, [False] * 5 + [True])
        self.assertEqual(s.best_epoch, 5)  # 微小提升也必须保存最佳
        self.assertEqual(s.outcome(True), "validation_score_plateau")

    def test_cumulative_gain_and_boundary(self):
        s, _ = self.run_scores([.1, .2, .201, .202])
        self.assertEqual(s.bad_epochs, 0)
        self.assertEqual(s.anchor, .202)

    def test_decline_not_stable(self):
        s, flags = self.run_scores([.1, .2, .19, .18, .17, .16])
        self.assertTrue(flags[-1])
        self.assertEqual(s.outcome(True), "no_improvement_with_regression")

    def test_zero_not_success(self):
        s, _ = self.run_scores([0.] * 6)
        self.assertEqual(s.outcome(True), "no_detection_progress")

    def test_fluctuating(self):
        s, _ = self.run_scores([.1, .2, .18, .20, .19, .20])
        self.assertEqual(s.outcome(True), "no_improvement_but_fluctuating")

    def test_resume(self):
        scores = [.1, .2, .201, .202, .201, .203, .202, .203]
        full, flags = self.run_scores(scores)
        partial, _ = self.run_scores(scores[:4])
        restored = StopState(**json.loads(json.dumps(partial.to_dict())))
        resumed_flags = [restored.update(i, scores[i]) for i in range(4, len(scores))]
        self.assertEqual(flags[4:], resumed_flags)
        self.assertEqual(full.to_dict(), restored.to_dict())

    def test_invalid_score_or_epoch(self):
        for epoch, score in [(1, .2), (0, float("nan")), (0, -1)]:
            with self.assertRaises(ValueError):
                StopState().update(epoch, score)


if __name__ == "__main__":
    unittest.main()
