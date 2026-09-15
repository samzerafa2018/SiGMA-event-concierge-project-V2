import os
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class ConversationEndToEndTests(unittest.TestCase):
    def test_availability_topic_and_itinerary_conversation(self) -> None:
        environment = {
            "LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": "",
            "LLM_MAX_ATTEMPTS": "1",
        }

        with patch.dict(os.environ, environment, clear=False):
            app = AppTest.from_file(
                APP_PATH,
                default_timeout=10,
            ).run()

            questions = (
                "I can only attend on Wednesday from 13:00 to 18:00.",
                "Which AI sessions can I attend?",
                "Build an agenda using only those sessions.",
            )

            for question in questions:
                app.chat_input[0].set_value(question).run()
                self.assertEqual(len(app.exception), 0)

        messages = app.session_state["messages"]
        availability_answer = messages[1]
        topic_answer = messages[3]
        itinerary_answer = messages[5]

        self.assertIn("Availability saved", availability_answer["text"])
        self.assertEqual(
            topic_answer["sources"],
            ["S014", "S015", "S033"],
        )
        self.assertEqual(
            itinerary_answer["sources"],
            ["S014", "S015", "S033"],
        )

        for source_id in ("S014", "S015", "S033"):
            self.assertIn(f"[{source_id}]", itinerary_answer["text"])


if __name__ == "__main__":
    unittest.main()
