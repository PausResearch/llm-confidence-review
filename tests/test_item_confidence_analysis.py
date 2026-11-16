import copy
import unittest

import numpy as np

from diagnostics.item_confidence_analysis import analyze, fit, metrics, paired_bootstrap, predict, validate_rows


def sample_rows():
    rng = np.random.default_rng(91)
    rows = []
    for split, n in (("calibration", 100), ("development", 32)):
        for i in range(n):
            p = float(rng.choice([.2, .4, .7, .9]))
            r = int(rng.integers(2))
            q = float(np.clip(p + rng.normal(0, .15), .02, .98))
            rows.append({"id": f"{split}-{i}", "domain": "a" if i % 2 else "b", "split": split,
                         "p": p, "r": r, "q": q, "q2": float(np.clip(q + rng.normal(0, .08), .02, .98)),
                         "y": int(rng.random() < q)})
    return rows


class ItemConfidenceAnalysisTests(unittest.TestCase):
    def test_development_labels_cannot_change_fits_or_predictions(self):
        original = sample_rows()
        changed = copy.deepcopy(original)
        for row in changed:
            if row["split"] == "development":
                row["y"] = 1 - row["y"]
        first, second = analyze(original), analyze(changed)
        self.assertEqual(first["models"], second["models"])
        for a, b in zip(first["predictions"], second["predictions"]):
            self.assertEqual(a["fitted_predictions"], b["fitted_predictions"])
        self.assertNotEqual(first["scores"], second["scores"])

    def test_evaluation_duplicates_and_unseen_domains_rejected(self):
        rows = sample_rows()
        for mutation in ("evaluation", "duplicate", "unknown_domain"):
            bad = copy.deepcopy(rows)
            if mutation == "evaluation":
                bad[-1]["split"] = "evaluation"
            elif mutation == "duplicate":
                bad[-1]["id"] = bad[0]["id"]
            else:
                bad[-1]["domain"] = "unseen"
            with self.assertRaises(ValueError):
                validate_rows(bad)
        with self.assertRaises(ValueError):
            fit(rows)

    def test_paired_bootstrap_preserves_domain_weights(self):
        # Within-domain constants imply zero bootstrap variance even though
        # between-domain means differ. Domain sizes must remain 3:1.
        result = paired_bootstrap([-.2, -.2, -.2, .2], ["a", "a", "a", "b"], 400)
        self.assertAlmostEqual(result["mean_difference"], -.1)
        np.testing.assert_allclose(result["ci95"], [-.1, -.1])
        zero = paired_bootstrap([0, 0, 0], ["a", "a", "b"], 400)
        self.assertEqual(zero["ci95"], [0., 0.])

    def test_reliability_interaction_handles_reliable_low_confidence(self):
        rows = []
        for p in (.2, .8):
            for r in (0, 1):
                probability = p if r else .5
                for i in range(100):
                    rows.append({"id": f"{p}-{r}-{i}", "domain": "a", "split": "calibration",
                                 "p": p, "r": r, "q": .5, "q2": .5, "y": int(i < 100 * probability)})
        fitted = fit(rows, ("r",))
        probe = [{"p": p, "r": r, "domain": "a"} for p, r in ((.2, 1), (.2, 0), (.8, 1), (.8, 0))]
        values = predict(fitted, probe)
        self.assertLess(values[0], values[1])
        self.assertGreater(values[2], values[3])

    def test_decision_boundary_and_fixed_coverage_ties(self):
        result = metrics([.8, .8, .79, .1], [1, 0, 1, 0], ["a", "b", "c", "d"])
        self.assertEqual(result["utility_secondary"]["accepted"], 2)
        self.assertEqual(result["utility_secondary"]["points"], -3)
        self.assertEqual(result["selective_accuracy"]["0.5"]["accepted"], 2)
        self.assertEqual(result["selective_accuracy"]["0.5"]["accuracy"], .5)

    def test_constant_reliability_does_not_create_spurious_signal(self):
        rows = sample_rows()
        train = [row for row in rows if row["split"] == "calibration"]
        test = [row for row in rows if row["split"] == "development"]
        for row in rows:
            row["r"] = 1
        plain, augmented = fit(train), fit(train, ("r",))
        np.testing.assert_allclose(predict(plain, test), predict(augmented, test), atol=1e-9)


if __name__ == "__main__":
    unittest.main()
