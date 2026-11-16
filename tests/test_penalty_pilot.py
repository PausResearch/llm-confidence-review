import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from auditstudy import penalty_pilot as amendment
from auditstudy import prompts, runner
from auditstudy.common import config, digest
from auditstudy.data import prepare
from auditstudy.science import summaries


class PenaltyTests(unittest.TestCase):
    def test_prompt_changes_only_payoff_and_threshold_and_hides_gold(self):
        cs = dict(summaries([.85, .95]), gold="SECRET", rates="SECRET")
        args = ("batch", [.85, .95], [None, None, .92], cs, 3, 10)
        original = prompts.decide(*args)
        modified = amendment.decision_prompt(*args)
        expected = original[1]["content"].replace("accepted wrong answers earn -4", "accepted wrong answers earn -9").replace("threshold is probability at least 0.8", "threshold is probability at least 0.9")
        self.assertEqual(modified[0], original[0])
        self.assertEqual(modified[1]["content"], expected)
        self.assertNotIn("SECRET", expected)

    def test_payoff_threshold_and_direct_probabilities(self):
        result = amendment.release([.899999, .9, .99, .7], [1, 1, 0, 0])
        self.assertEqual(result["points"], -8)
        self.assertEqual(result["accepted"], 2)
        self.assertEqual(result["wrong_released"], 1)
        self.assertEqual(result["coverage"], .5)
        self.assertAlmostEqual(amendment.THRESHOLD,
                               -amendment.REWARD_WRONG / (amendment.REWARD_CORRECT - amendment.REWARD_WRONG))

    def test_useful_harmful_ties_and_missing_are_distinct(self):
        def row(gain, action):
            return {"gross_audit_gain": gain, "trust": {"points": 0}, "actions": {"3": {"M": action}}}
        result = amendment.choice_metrics([row(7, "AUDIT"), row(7, "TRUST"), row(0, "AUDIT"),
                                           row(0, "TRUST"), row(3, "AUDIT"), row(3, "TRUST"), row(7, None)], "M", 3)
        self.assertEqual(result["useful_taken"], 1)
        self.assertEqual(result["useful_missed"], 1)
        self.assertEqual(result["harmful_taken"], 1)
        self.assertEqual(result["harmful_avoided"], 1)
        self.assertEqual(result["ties_audited"], 1)
        self.assertEqual(result["ties_trusted"], 1)
        self.assertEqual(result["missing"], 1)
        self.assertEqual(result["correct_non_tie_choices"], 2)
        self.assertEqual(result["non_tie_batches"], 4)
        self.assertAlmostEqual(result["mean_points"], 1 / 6)
        self.assertAlmostEqual(result["mean_realized_regret"], 7 / 6)

    def test_decision_only_run_preserves_parent_and_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, parent, output = root / "data", root / "parent", root / "amended"
            c = config()
            c.update(domains=["biology", "economics"], calibration_per_domain=20,
                     development_per_domain=20, evaluation_per_domain=20, batch_size=10, workers=2)
            with patch.object(runner, "ROOT", root), patch.object(amendment, "ROOT", root), contextlib.redirect_stdout(io.StringIO()):
                prepare(c, data, "mock")
                runner.run(c, data, parent, "mock")
                before = digest(amendment.task_rows(parent / "tasks.sqlite"))
                source = amendment.load_parent(parent)
                jobs = amendment.build_jobs(source)
                self.assertEqual(len(jobs), 36)
                self.assertTrue(all(j[1] == "decide" and "development" in j[0] for j in jobs))
                updated = amendment.initialize(source, output, jobs)
                self.assertEqual(updated["threshold"], .9)
                self.assertEqual(updated["reward_wrong"], -9)
                run = amendment.run_decisions(source, output, updated, jobs, 1)
                self.assertEqual(run["valid_this_run"], 36)
                self.assertEqual(run["errors"], [])
                repeated = amendment.run_decisions(source, output, updated, jobs, 1)
                self.assertEqual(repeated["valid_this_run"], 0)
                report = amendment.score(source, output, jobs)
                self.assertEqual(report["checks"]["evaluation_tasks"], 0)
                self.assertEqual(report["checks"]["new_task_kinds"], ["decide"])
                self.assertEqual(report["by_cost"]["3"]["paired_batches"], 4)
                self.assertIn("R_direct", report["by_cost"]["3"]["baseline_mean_points"])
                self.assertEqual(digest(amendment.task_rows(parent / "tasks.sqlite")), before)
                self.assertFalse((parent / "frozen.json").exists())
                self.assertFalse((output / "frozen.json").exists())
                # Mutation cannot be silently resumed under the same amendment.
                source["config"] = dict(source["config"], temperature=.5)
                with self.assertRaises(ValueError):
                    amendment.initialize(source, output, jobs)


if __name__ == "__main__":
    unittest.main()
