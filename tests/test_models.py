import copy
import json
import tempfile
import unittest
from pathlib import Path

from concierge.models import DEFAULT_DATA_PATH, load_agenda


class AgendaModelValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.valid_data = json.loads(
            DEFAULT_DATA_PATH.read_text(encoding="utf-8")
        )

    def assert_invalid_agenda(
        self,
        agenda_data: dict,
        expected_message: str,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            agenda_path = Path(temporary_directory) / "agenda.json"
            agenda_path.write_text(
                json.dumps(agenda_data),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, expected_message):
                load_agenda(agenda_path)

    def test_missing_session_id_is_rejected(self) -> None:
        agenda_data = copy.deepcopy(self.valid_data)
        del agenda_data["sessions"][0]["id"]

        self.assert_invalid_agenda(agenda_data, "Field required")

    def test_impossible_session_date_is_rejected(self) -> None:
        agenda_data = copy.deepcopy(self.valid_data)
        agenda_data["sessions"][0]["day"] = "Tuesday 2026-99-99"

        self.assert_invalid_agenda(agenda_data, "valid ISO date")

    def test_impossible_session_time_is_rejected(self) -> None:
        agenda_data = copy.deepcopy(self.valid_data)
        agenda_data["sessions"][0]["start"] = "25:00"

        self.assert_invalid_agenda(agenda_data, "valid 24-hour times")

    def test_session_end_before_start_is_rejected(self) -> None:
        agenda_data = copy.deepcopy(self.valid_data)
        agenda_data["sessions"][0]["start"] = "15:00"
        agenda_data["sessions"][0]["end"] = "14:00"

        self.assert_invalid_agenda(agenda_data, "later than its start")

    def test_duplicate_session_ids_are_rejected(self) -> None:
        agenda_data = copy.deepcopy(self.valid_data)
        agenda_data["sessions"][1]["id"] = agenda_data["sessions"][0]["id"]

        self.assert_invalid_agenda(agenda_data, "Duplicate session IDs")

    def test_duplicate_exhibitor_ids_are_rejected(self) -> None:
        agenda_data = copy.deepcopy(self.valid_data)
        agenda_data["exhibitors"][1]["id"] = agenda_data["exhibitors"][0]["id"]

        self.assert_invalid_agenda(agenda_data, "Duplicate exhibitor IDs")


if __name__ == "__main__":
    unittest.main()