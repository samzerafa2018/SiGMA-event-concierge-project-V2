import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from concierge.llm import (
    GeminiProvider,
    LLMError,
    RetryingProvider,
)


class SequenceProvider:
    def __init__(self, outcomes: list[str | LLMError]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        outcome = self.outcomes[self.calls]
        self.calls += 1

        if isinstance(outcome, LLMError):
            raise outcome

        return outcome


class RetryingProviderTests(unittest.TestCase):
    def test_temporary_failure_is_retried_once(self) -> None:
        provider = SequenceProvider(
            [
                LLMError("Temporary connection failure", retryable=True),
                "Recovered answer",
            ]
        )
        retrying_provider = RetryingProvider(
            provider,
            max_attempts=2,
            delay_seconds=0,
        )

        answer = retrying_provider.generate("system", "question")

        self.assertEqual(answer, "Recovered answer")
        self.assertEqual(provider.calls, 2)

    def test_non_retryable_failure_is_not_retried(self) -> None:
        provider = SequenceProvider(
            [LLMError("Invalid response", retryable=False)]
        )
        retrying_provider = RetryingProvider(
            provider,
            max_attempts=3,
            delay_seconds=0,
        )

        with self.assertRaisesRegex(LLMError, "Invalid response"):
            retrying_provider.generate("system", "question")

        self.assertEqual(provider.calls, 1)

    def test_final_temporary_failure_is_raised(self) -> None:
        provider = SequenceProvider(
            [
                LLMError("Temporary failure 1", retryable=True),
                LLMError("Temporary failure 2", retryable=True),
            ]
        )
        retrying_provider = RetryingProvider(
            provider,
            max_attempts=2,
            delay_seconds=0,
        )

        with self.assertRaisesRegex(LLMError, "Temporary failure 2"):
            retrying_provider.generate("system", "question")

        self.assertEqual(provider.calls, 2)

    def test_invalid_retry_configuration_is_rejected(self) -> None:
        provider = SequenceProvider(["Unused answer"])

        with self.assertRaisesRegex(ValueError, "max_attempts"):
            RetryingProvider(provider, max_attempts=0)

        with self.assertRaisesRegex(ValueError, "delay_seconds"):
            RetryingProvider(provider, delay_seconds=-1)


class FakeResponse:
    def __init__(self, data: object) -> None:
        self.data = data

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.data


def fake_requests_module(response_data: object) -> SimpleNamespace:
    request_exception = type("RequestException", (Exception,), {})
    timeout = type("Timeout", (request_exception,), {})
    connection_error = type(
        "ConnectionError",
        (request_exception,),
        {},
    )
    http_error = type("HTTPError", (request_exception,), {})

    return SimpleNamespace(
        post=Mock(return_value=FakeResponse(response_data)),
        Timeout=timeout,
        ConnectionError=connection_error,
        HTTPError=http_error,
        RequestException=request_exception,
    )


class GeminiProviderTests(unittest.TestCase):
    def test_model_output_is_returned(self) -> None:
        fake_requests = fake_requests_module(
            {
                "steps": [
                    {
                        "type": "model_output",
                        "content": [
                            {"type": "text", "text": "Grounded "},
                            {"type": "text", "text": "answer. [S017]"},
                        ],
                    }
                ]
            }
        )
        provider = GeminiProvider(api_key="test-key")

        with patch.dict(sys.modules, {"requests": fake_requests}):
            answer = provider.generate("system rules", "question")

        self.assertEqual(answer, "Grounded answer. [S017]")

    def test_request_uses_safe_fast_configuration(self) -> None:
        fake_requests = fake_requests_module(
            {"output_text": "Answer. [S017]"}
        )
        provider = GeminiProvider(api_key="secret-test-key")

        with patch.dict(sys.modules, {"requests": fake_requests}):
            provider.generate("system rules", "question")

        _, call_kwargs = fake_requests.post.call_args
        payload = call_kwargs["json"]

        self.assertEqual(
            call_kwargs["headers"]["x-goog-api-key"],
            "secret-test-key",
        )
        self.assertNotIn("secret-test-key", fake_requests.post.call_args.args[0])
        self.assertEqual(payload["model"], "gemini-3.7-flash")
        self.assertEqual(payload["generation_config"]["thinking_level"], "low")
        self.assertFalse(payload["store"])

    def test_missing_api_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(LLMError, "GEMINI_API_KEY"):
            GeminiProvider(api_key="  ")

    def test_empty_response_is_retryable(self) -> None:
        fake_requests = fake_requests_module({"steps": []})
        provider = GeminiProvider(api_key="test-key")

        with patch.dict(sys.modules, {"requests": fake_requests}):
            with self.assertRaisesRegex(LLMError, "empty") as context:
                provider.generate("system rules", "question")

        self.assertTrue(context.exception.retryable)


if __name__ == "__main__":
    unittest.main()
