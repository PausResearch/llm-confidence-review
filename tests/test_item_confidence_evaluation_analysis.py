import copy
import unittest
from unittest.mock import patch

import numpy as np

from auditstudy.common import digest
from diagnostics.item_confidence_analysis import MODELS, fit, predict
from diagnostics.item_confidence_evaluation_analysis import analyze_evaluation, validate_evaluation


def fixture():
    rng = np.random.default_rng(716)
    train, evaluation = [], []
    for split, size, destination in (("calibration", 480, train), ("evaluation", 40, evaluation)):
        for i in range(size):
            p = float(rng.choice([.2, .5, .75, .95]))
            q = float(np.clip(p + rng.normal(0, .15), .01, .99))
            destination.append({"id": f"{split}-{i}", "split": split, "domain": "a" if i % 2 else "b",
                                "p": p, "q": q, "q2": float(np.clip(q + rng.normal(0, .1), .01, .99)),
                                "r": int(rng.integers(2)), "y": int(rng.random() < q)})
    return train, evaluation, {name: fit(train, signals) for name, signals in MODELS.items()}


class EvaluationAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.rows, cls.models = fixture()

    def test_no_fit_no_mutation_and_predictions_match_original_maps(self):
        before = digest(self.models)
        original_rows = copy.deepcopy(self.rows)
        with patch("diagnostics.item_confidence_analysis.fit", side_effect=AssertionError("Must not fit")):
            result = analyze_evaluation(self.rows, self.models)
        self.assertEqual(before, digest(self.models))
        self.assertEqual(original_rows, self.rows)
        self.assertEqual(result["frozen_models_hash"], before)
        self.assertFalse(result["fit_performed"])
        ordered = sorted(self.rows, key=lambda row: row["id"])
        for name, spec in self.models.items():
            np.testing.assert_array_equal(predict(spec, ordered),
                [row["fitted_predictions"][name] for row in result["predictions"]])

    def test_labels_change_scores_not_frozen_predictions(self):
        changed = copy.deepcopy(self.rows)
        for row in changed:
            row["y"] = 1 - row["y"]
        a, b = analyze_evaluation(self.rows, self.models), analyze_evaluation(changed, self.models)
        self.assertEqual(a["models"], b["models"])
        self.assertEqual([r["fitted_predictions"] for r in a["predictions"]],
                         [r["fitted_predictions"] for r in b["predictions"]])
        self.assertNotEqual(a["scores"], b["scores"])

    def test_eval_only_unique_ids_and_seen_domains(self):
        self.assertFalse({r["id"] for r in self.train} & {r["id"] for r in self.rows})
        for kind in ("calibration", "development", "duplicate", "unseen"):
            bad = copy.deepcopy(self.rows)
            if kind in ("calibration", "development"):
                bad[0]["split"] = kind
            elif kind == "duplicate":
                bad[0]["id"] = bad[1]["id"]
            else:
                bad[0]["domain"] = "unseen"
            with self.assertRaises(ValueError):
                validate_evaluation(bad, self.models)

    def test_invalid_models_rejected(self):
        for kind in ("n_fit", "signal", "clip", "scale", "coefficients"):
            bad = copy.deepcopy(self.models)
            if kind == "n_fit":
                bad["p"]["n_fit"] = 160
            elif kind == "signal":
                bad["p"]["signals"] = ["q"]
            elif kind == "clip":
                bad["p"]["clip"] = .01
            elif kind == "scale":
                bad["p"]["scalers"]["p"]["scale"] = 0
            else:
                bad["p"]["coefficients"][0] = float("nan")
            with self.assertRaises(ValueError):
                validate_evaluation(self.rows, bad)

    def test_contrasts_recompute_directly_and_conditional_intervals(self):
        result = analyze_evaluation(self.rows, self.models)
        for c in result["contrasts"].values():
            a, b = c["first_model"], c["second_model"]
            self.assertAlmostEqual(c["brier"]["mean_difference"],
                                   result["scores"][a]["brier"] - result["scores"][b]["brier"])
            self.assertTrue(c["brier"]["conditional_on_fitted_models"])
            self.assertEqual(c["brier"]["bootstrap_repetitions"], 5000)
        gate = result["interpretation_gate"]
        self.assertEqual(gate["specificity_benefit_supported_in_fixed_sequence"],
                         gate["primary_benefit_supported"] and result["contrasts"]["matched_two_calls"]["brier"]["ci95"][1] < 0)
        self.assertTrue(gate["no_equivalence_or_absence_claim"])


if __name__ == "__main__":
    unittest.main()
