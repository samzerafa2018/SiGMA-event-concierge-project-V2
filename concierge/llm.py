from __future__ import annotations

import time
from typing import Protocol


class LLMError(Exception):
    """Raised when an LLM provider cannot generate a response."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable


class LLMProvider(Protocol):
    """Small interface that lets the application swap model providers."""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Return a complete text response from the model."""
        ...


class RetryingProvider:
    """Retry a provider only when it reports a temporary failure."""

    def __init__(
        self,
        provider: LLMProvider,
        max_attempts: int = 2,
        delay_seconds: float = 0.5,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        if delay_seconds < 0:
            raise ValueError("delay_seconds cannot be negative")

        self.provider = provider
        self.max_attempts = max_attempts
        self.delay_seconds = delay_seconds

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self.provider.generate(
                    system_prompt,
                    user_prompt,
                )
            except LLMError as error:
                if not error.retryable or attempt == self.max_attempts:
                    raise

                if self.delay_seconds:
                    time.sleep(self.delay_seconds)

        raise AssertionError("Retry loop ended unexpectedly.")


class GeminiProvider:
    """Cloud Gemini implementation using Google's REST API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.7-flash",
        base_url: str = (
            "https://generativelanguage.googleapis.com/v1beta"
        ),
        timeout_seconds: int = 45,
        thinking_level: str = "low",
    ) -> None:
        api_key = api_key.strip()

        if not api_key:
            raise LLMError(
                "GEMINI_API_KEY is not configured. Add your Gemini API "
                "key to the local .env file and restart the app."
            )

        allowed_thinking_levels = {
            "minimal",
            "low",
            "medium",
            "high",
        }
        thinking_level = thinking_level.strip().lower()

        if thinking_level not in allowed_thinking_levels:
            raise ValueError(
                "thinking_level must be minimal, low, medium, or high"
            )

        self.api_key = api_key
        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.thinking_level = thinking_level

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        try:
            import requests
        except ImportError as error:
            raise LLMError(
                "The requests package is required to call Gemini. "
                "Install the project requirements and try again."
            ) from error

        payload = {
            "model": self.model,
            "system_instruction": system_prompt,
            "input": user_prompt,
            # Each concierge call is independent. The application already
            # manages its own availability state and does not need Google to
            # retain interaction history.
            "store": False,
            "generation_config": {
                "thinking_level": self.thinking_level,
                "max_output_tokens": 768,
            },
        }

        try:
            response = requests.post(
                f"{self.base_url}/interactions",
                headers={
                    "x-goog-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.Timeout as error:
            # A second full timeout would make the interface slower. The
            # answering layer can return its verified deterministic fallback.
            raise LLMError(
                "Gemini took too long to respond. A verified agenda "
                "summary will be used when possible."
            ) from error
        except requests.ConnectionError as error:
            raise LLMError(
                "Could not reach Gemini. Check the internet connection and "
                "try again.",
                retryable=True,
            ) from error
        except requests.HTTPError as error:
            status_code = (
                error.response.status_code
                if error.response is not None
                else None
            )
            retryable = (
                status_code == 429
                or (
                    status_code is not None
                    and status_code >= 500
                )
            )

            if status_code in {401, 403}:
                message = (
                    "Gemini rejected the API key. Check GEMINI_API_KEY in "
                    "the local .env file."
                )
            elif status_code == 429:
                message = (
                    "Gemini's free-tier limit is temporarily busy or has "
                    "been reached. Try again shortly."
                )
            else:
                message = (
                    "Gemini returned an HTTP error while generating an "
                    "answer."
                )

            raise LLMError(
                message,
                retryable=retryable,
            ) from error
        except requests.RequestException as error:
            raise LLMError(
                "Gemini request failed unexpectedly.",
                retryable=True,
            ) from error

        try:
            response_data = response.json()
        except ValueError as error:
            raise LLMError(
                "Gemini returned an unexpected response format."
            ) from error

        answer = self._extract_answer(response_data)

        if not answer:
            raise LLMError(
                "Gemini returned an empty response.",
                retryable=True,
            )

        return answer

    @staticmethod
    def _extract_answer(response_data: object) -> str:
        """Extract text from the final model-output step."""
        if not isinstance(response_data, dict):
            return ""

        output_text = response_data.get("output_text")

        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        steps = response_data.get("steps")

        if not isinstance(steps, list):
            return ""

        for step in reversed(steps):
            if (
                not isinstance(step, dict)
                or step.get("type") != "model_output"
            ):
                continue

            content = step.get("content")

            if not isinstance(content, list):
                continue

            text_parts = [
                part["text"]
                for part in content
                if (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                )
            ]
            answer = "".join(text_parts).strip()

            if answer:
                return answer

        return ""


class OllamaProvider:
    """Local Ollama implementation of the LLM provider interface."""

    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        try:
            import requests
        except ImportError as error:
            raise LLMError(
                "The requests package is required to call Ollama. "
                "Install the project requirements and try again."
            ) from error

        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": user_prompt,
            "think": False,
            "stream": False,
            "options": {
                # Low temperature improves factual consistency and citations.
                "temperature": 0.1,
                # Only retrieved evidence is sent, so 8K is sufficient.
                "num_ctx": 8192,
                # Allow complete lists without encouraging excessive output.
                "num_predict": 768,
            },
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.Timeout as error:
            # Do not repeat a full timeout. The answering layer can immediately
            # return its verified deterministic fallback instead.
            raise LLMError(
                "Ollama took too long to respond. Try a shorter question or "
                "wait for the local model to finish loading."
            ) from error
        except requests.ConnectionError as error:
            raise LLMError(
                "Could not reach Ollama. Make sure Ollama is installed, "
                "running, and the selected model has been downloaded.",
                retryable=True,
            ) from error
        except requests.HTTPError as error:
            status_code = (
                error.response.status_code
                if error.response is not None
                else None
            )
            retryable = (
                status_code == 429
                or (
                    status_code is not None
                    and status_code >= 500
                )
            )

            raise LLMError(
                "Ollama returned an HTTP error while generating an answer.",
                retryable=retryable,
            ) from error
        except requests.RequestException as error:
            raise LLMError(
                "Ollama request failed unexpectedly.",
                retryable=True,
            ) from error

        try:
            answer = response.json()["response"].strip()
        except (KeyError, ValueError) as error:
            raise LLMError(
                "Ollama returned an unexpected response format."
            ) from error

        if not answer:
            raise LLMError(
                "Ollama returned an empty response.",
                retryable=True,
            )

        return answer


class MockProvider:
    """
    Predictable provider for tests or demos without Ollama.

    It satisfies the same interface as OllamaProvider, so the rest of the
    application does not need to know which provider it is using.
    """

    def __init__(
        self,
        response: str = "Mock response.",
    ) -> None:
        self.response = response

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        return self.response
