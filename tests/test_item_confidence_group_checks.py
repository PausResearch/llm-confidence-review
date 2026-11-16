import copy
import unittest

import numpy as np

from diagnostics.item_confidence_group_checks import analyze_groups, band_index, cell_summary


def fixture():
    records = []
    for domain in ("a", "b"):
        for j, p in enumerate((0, .599, .6, .799, .8, 1)):
            records.append({"id": domain + str(j), "domain": domain, "split": "evaluation",
                            "p": p, "y": 0, "r": j % 2})
    predictions = [{**r, "fitted_predictions": {"p": .7, "p_r": .2, "p_q": .6,
                                               "p_q_r": .3, "p_q_q2": .5}} for r in records]
    return records, predictions


class GroupChecksTests(unittest.TestCase):
    def test_bands_boundaries_and_counts(self):
        self.assertEqual([band_index(x) for x in (0, .599, .6, .799, .8, 1)], [0, 0, 1, 1, 2, 2])
        for value in (-.001, 1.001, float("nan")):
            with self.assertRaises(ValueError):
                band_index(value)
        records, predictions = fixture()
        result = analyze_groups(records, predictions, repetitions=100)
        self.assertEqual(len(result["calibration_cells"]), 12)
        self.assertEqual(sum(c["n"] for c in result["calibration_cells"]), len(records))
        self.assertTrue(all(c["small_group"] for c in result["calibration_cells"]))

    def test_probabilities_and_outcomes_resampled_as_pairs(self):
        rows = [{"p": 0, "y": 0}, {"p": 1, "y": 1}] * 15
        summary, samples = cell_summary(rows, np.random.default_rng(1), 200)
        self.assertEqual(summary["signed_gap_ci95"], [0., 0.])
        self.assertTrue(np.all(samples == 0))
        self.assertFalse(summary["small_group"])

    def test_domain_paired_loss_difference_and_inputs_preserved(self):
        records, predictions = fixture()
        before = copy.deepcopy((records, predictions))
        result = analyze_groups(records, predictions, repetitions=100)
        self.assertEqual((records, predictions), before)
        for domain in result["by_domain"]:
            value = domain["contrasts"]["matched_one_call"]
            self.assertAlmostEqual(value["mean_difference"], -.32)
            self.assertTrue(np.allclose(value["ci95"], [-.32, -.32]))

    def test_support_and_label_disagreement_rejected(self):
        records, predictions = fixture()
        with self.assertRaises(ValueError):
            analyze_groups(records, predictions[:-1])
        predictions[0]["y"] = 1
        with self.assertRaises(ValueError):
            analyze_groups(records, predictions)
        predictions[0]["y"] = 0
        records[0]["split"] = "calibration"
        with self.assertRaises(ValueError):
            analyze_groups(records, predictions)


if __name__ == "__main__":
    unittest.main()
