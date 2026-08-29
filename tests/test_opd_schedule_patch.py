from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from patches.apply_patch import PATCHES, apply_one
from reasonsql.opd.schedule import OPDSchedule


class ScheduleTests(unittest.TestCase):
    def test_constant_linear_and_cosine(self) -> None:
        constant = OPDSchedule(start=0.7, end=0.1, kind="constant")
        self.assertEqual(constant.coefficient(50, 100), 0.7)

        linear = OPDSchedule(start=1.0, end=0.2, kind="linear")
        self.assertAlmostEqual(linear.coefficient(0, 100), 1.0)
        self.assertAlmostEqual(linear.coefficient(50, 100), 0.6)
        self.assertAlmostEqual(linear.coefficient(100, 100), 0.2)

        cosine = OPDSchedule(start=1.0, end=0.0, kind="cosine")
        self.assertAlmostEqual(cosine.coefficient(0, 100), 1.0)
        self.assertAlmostEqual(cosine.coefficient(50, 100), 0.5)
        self.assertAlmostEqual(cosine.coefficient(100, 100), 0.0)

    def test_warmup_delays_decay(self) -> None:
        schedule = OPDSchedule(start=1.0, end=0.0, kind="linear", warmup_steps=10)
        self.assertEqual(schedule.coefficient(5, 100), 1.0)
        self.assertAlmostEqual(schedule.coefficient(55, 100), 0.5)


class PatchTests(unittest.TestCase):
    def test_patch_anchors_and_idempotence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ray_path = root / PATCHES[0].relative_path
            loss_path = root / PATCHES[1].relative_path
            ray_path.parent.mkdir(parents=True)
            loss_path.parent.mkdir(parents=True)
            ray_path.write_text(
                "def f(self, batch):\n"
                "    batch_td = batch.to_tensordict()\n"
                "    return batch_td\n",
                encoding="utf-8",
            )
            loss_path.write_text(
                "def f(data, policy_loss, distill_loss, distillation_loss_coef, policy_metrics):\n"
                "    policy_loss += distill_loss * distillation_loss_coef\n"
                "    return policy_loss\n",
                encoding="utf-8",
            )
            self.assertTrue(apply_one(root, PATCHES[0], dry_run=False))
            self.assertTrue(apply_one(root, PATCHES[1], dry_run=False))
            self.assertIn("ReasonSQL OPD schedule patch", ray_path.read_text(encoding="utf-8"))
            self.assertIn("scheduled_coef", loss_path.read_text(encoding="utf-8"))
            self.assertTrue(apply_one(root, PATCHES[0], dry_run=False))
            self.assertTrue(apply_one(root, PATCHES[1], dry_run=False))


if __name__ == "__main__":
    unittest.main()

