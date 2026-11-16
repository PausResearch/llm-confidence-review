import copy
import unittest
from unittest.mock import patch

import numpy as np

from diagnostics.item_confidence_refit_sensitivity import (
    MODELS, replicate, stratified_indices, summarize, validate_inputs,
)


def fixture():
    out = []
    for split, n in (("calibration", 24), ("evaluation", 40)):
        rows = []
        for i in range(n):
            rows.append({"id": split + str(i), "split": split, "domain": "a" if i % 2 else "b",
                         "p": [.3, .6, .9][i % 3], "q": [.2, .7, .8][i % 3], "q2": [.4, .65, .85][i % 3],
                         "r": i % 2, "y": int(i % 3 != 0)})
        out.append(rows)
    return out


class RefitSensitivityTests(unittest.TestCase):
    def test_stratified_draw_preserves_sizes_and_is_reproducible(self):
        train, _ = fixture()
        a = stratified_indices(train, np.random.default_rng(7))
        b = stratified_indices(train, np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(a), len(train))
        self.assertEqual(sum(train[i]["domain"] == "a" for i in a), 12)
        self.assertTrue(len(set(a)) < len(a))

    def test_shared_resamples_and_only_train_rows_fit(self):
        train, evaluation = fixture()
        before = copy.deepcopy((train, evaluation))
        fit_ids, eval_ids = [], []
        def fake_fit(rows, signals):
            self.assertTrue(all(r["split"] == "calibration" for r in rows))
            fit_ids.append([r["id"] for r in rows])
            return {"signals": signals}
        def fake_predict(spec, rows):
            self.assertTrue(all(r["split"] == "evaluation" for r in rows))
            eval_ids.append([r["id"] for r in rows])
            return np.array([r["p"] for r in rows])
        with patch("diagnostics.item_confidence_refit_sensitivity.fit", side_effect=fake_fit), \
             patch("diagnostics.item_confidence_refit_sensitivity.predict", side_effect=fake_predict):
            result = replicate(train, evaluation, 11)
        self.assertEqual(len(fit_ids), len(MODELS))
        self.assertTrue(all(ids == fit_ids[0] for ids in fit_ids))
        self.assertTrue(all(ids == eval_ids[0] for ids in eval_ids))
        self.assertFalse(set(fit_ids[0]) & set(eval_ids[0]))
        self.assertEqual(before, (train, evaluation))
        self.assertTrue(all(c["brier"] == 0 for c in result["contrasts"].values()))

    def test_inputs_reject_overlap_and_mixed_splits(self):
        train, evaluation = fixture()
        validate_inputs(train, evaluation)
        bad = copy.deepcopy(evaluation)
        bad[0]["id"] = train[0]["id"]
        with self.assertRaises(ValueError):
            validate_inputs(train, bad)
        bad = copy.deepcopy(train)
        bad[0]["split"] = "evaluation"
        with self.assertRaises(ValueError):
            validate_inputs(bad, evaluation)

    def test_real_refit_reproducible_and_source_unchanged(self):
        train, evaluation = fixture()
        before = copy.deepcopy((train, evaluation))
        a = replicate(train, evaluation, 5)
        b = replicate(train, evaluation, 5)
        self.assertEqual(a, b)
        self.assertEqual(before, (train, evaluation))


if __name__ == "__main__":
    unittest.main()
