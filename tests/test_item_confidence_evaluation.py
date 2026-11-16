import copy
import unittest

from auditstudy.prompts import generate
from diagnostics.item_confidence import messages_for
from diagnostics.item_confidence_evaluation import generation_jobs, review_jobs, review_messages


class EvaluationPromptTests(unittest.TestCase):
    def setUp(self):
        self.item = {"id": "e", "domain": "biology", "split": "evaluation", "question": "Example?",
                     "options": ["one", "two"], "gold": "SECRET_NEVER_SEND"}
        self.answer = {"answer": "A", "confidence": .8, "correct": "SECRET_NEVER_SEND"}

    def test_adapter_identical_prompt_and_original_row_preserved(self):
        before = copy.deepcopy(self.item)
        for arm in ("M", "R", "R2"):
            actual = review_messages(self.item, self.answer, arm)
            expected = messages_for(dict(self.item, split="development"), self.answer, arm)
            self.assertEqual(actual, expected)
            self.assertNotIn("SECRET", str(actual))
        self.assertEqual(self.item, before)
        with self.assertRaises(ValueError):
            review_messages(dict(self.item, split="development"), self.answer, "M")

    def test_generation_matches_original_and_review_requests_distinct(self):
        jobs = generation_jobs({"e": self.item})
        self.assertEqual(jobs[0][2], generate(self.item))
        self.assertEqual(jobs[0][4], 256)
        self.assertNotIn("SECRET", str(jobs))
        reviews, missing = review_jobs({"e": self.item}, {"generate:e": {"status": "done", "result": self.answer}})
        self.assertFalse(missing)
        self.assertEqual(len({j[0] for j in reviews}), 3)
        by_arm = {j[0].split(":")[-1]: j for j in reviews}
        self.assertEqual(by_arm["R"][2], by_arm["R2"][2])
        self.assertTrue(all(j[4] == 128 for j in reviews))

    def test_invalid_generation_not_reviewed_or_replaced(self):
        jobs, missing = review_jobs({"e": self.item}, {"generate:e": {"status": "invalid", "result": None}})
        self.assertEqual(jobs, [])
        self.assertEqual(missing, ["e"])


if __name__ == "__main__":
    unittest.main()
