import unittest

from diagnostics.item_confidence import messages_for, validate


class ItemConfidencePromptTests(unittest.TestCase):
    def setUp(self):
        self.item = {"id": "x", "domain": "biology", "split": "development",
                     "question": "Example?", "options": ["one", "two"], "gold": "NEVER_SEND_GOLD"}
        self.answer = {"answer": "A", "confidence": .6, "correct": "NEVER_SEND_CORRECTNESS"}

    def test_whitelist_no_labels_and_matched_frozen_record(self):
        m = messages_for(self.item, self.answer, "M")
        r = messages_for(self.item, self.answer, "R")
        self.assertEqual(m[0], r[0])
        self.assertEqual(m[1]["content"].split("\n")[-1], r[1]["content"].split("\n")[-1])
        self.assertNotIn("NEVER_SEND", str(m) + str(r))
        self.assertIn("A low probability can be reliable", m[1]["content"])

    def test_repeat_is_identical_prompt_and_eval_forbidden(self):
        self.assertEqual(messages_for(self.item, self.answer, "R"), messages_for(self.item, self.answer, "R2"))
        self.item["split"] = "evaluation"
        with self.assertRaises(ValueError):
            messages_for(self.item, self.answer, "M")

    def test_new_schema_strict_original_schema_preserved(self):
        self.assertEqual(validate("item_reliability", {"reliability": "LOW"}, {}), {"reliability": "LOW"})
        for bad in ({"reliability": "maybe"}, {"reliability": "HIGH", "p": .8}, None):
            with self.assertRaises(ValueError):
                validate("item_reliability", bad, {})
        self.assertEqual(validate("review_R", {"revised_probabilities": [{"id": "x", "p": .7}]}, {"ids": ["x"]}),
                         {"revised_probabilities": [{"id": "x", "p": .7}]})


if __name__ == "__main__":
    unittest.main()
