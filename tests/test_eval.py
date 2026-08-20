import unittest

from concierge.models import load_agenda
from concierge.retrieval import retrieve
from mini_eval import EVALUATION_CASES


class RetrievalEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.agenda = load_agenda()

    def test_retrieval_evaluation_cases(self) -> None:
        for case in EVALUATION_CASES:
            with self.subTest(question=case.question):
                items = retrieve(self.agenda, case.question)
                actual_ids = {item.id for item in items}

                self.assertTrue(
                    case.required_ids.issubset(actual_ids),
                    msg=(
                        f"Expected {sorted(case.required_ids)} for "
                        f"{case.question!r}, "
                        f"but retrieved {sorted(actual_ids)}"
                    ),
                )
                self.assertFalse(
                    case.forbidden_ids & actual_ids,
                    msg=(
                        f"Retrieved forbidden IDs "
                        f"{sorted(case.forbidden_ids & actual_ids)} for "
                        f"{case.question!r}"
                    ),
                )

                if case.exact_ids:
                    self.assertEqual(
                        actual_ids,
                        set(case.required_ids),
                        msg=(
                            f"Expected exactly "
                            f"{sorted(case.required_ids)} for "
                            f"{case.question!r}, but retrieved "
                            f"{sorted(actual_ids)}"
                        ),
                    )


if __name__ == "__main__":
    unittest.main()