import os
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class ApplicationSmokeTests(unittest.TestCase):
    def build_app(self) -> AppTest:
        environment = {
            "LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": "",
            "LLM_MAX_ATTEMPTS": "1",
        }

        with patch.dict(os.environ, environment, clear=False):
            return AppTest.from_file(
                APP_PATH,
                default_timeout=10,
            ).run()

    def test_application_starts_without_an_exception(self) -> None:
        app = self.build_app()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.chat_input), 1)
        self.assertGreaterEqual(len(app.button), 1)

    def test_exact_local_lookup_works_without_model_credentials(self) -> None:
        app = self.build_app()

        app.chat_input[0].set_value(
            "When and where is session S017?"
        ).run()

        self.assertEqual(len(app.exception), 0)
        assistant_message = app.session_state["messages"][-1]
        self.assertEqual(assistant_message["role"], "assistant")
        self.assertEqual(assistant_message["sources"], ["S017"])
        self.assertIn("14:00", assistant_message["text"])
        self.assertIn("Harbour Room", assistant_message["text"])


if __name__ == "__main__":
    unittest.main()
