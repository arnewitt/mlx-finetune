"""OpenAI-compatible API teacher.

Works with any OpenAI-compatible endpoint:
  - OpenAI           (gpt-4o, o3, ...)
  - NVIDIA NIM cloud (base_url="https://integrate.api.nvidia.com/v1", api_key via NVIDIA_API_KEY)
  - NVIDIA NIM local (base_url="http://localhost:8000/v1", api_key="not-used")
  - Ollama           (base_url="http://localhost:11434/v1", api_key="ollama")
  - LM Studio        (base_url="http://localhost:1234/v1",  api_key="lm-studio")
  - vLLM             (base_url="http://localhost:8000/v1",  api_key="none")
  - Together AI, Groq, Fireworks, etc.
"""

from __future__ import annotations

import os
import time

from finetune.teacher.base import TeacherBackend
from finetune.config import TeacherConfig


class OpenAICompatibleTeacher(TeacherBackend):
    """Calls any OpenAI-compatible Chat Completions endpoint as the teacher.

    Supports automatic retry with exponential back-off on rate-limit errors.
    The `api_key` and `base_url` fields in `TeacherConfig` allow routing to
    local servers (Ollama, LM Studio, vLLM) without code changes.
    """

    def __init__(
        self,
        cfg: TeacherConfig,
        examples: list[dict[str, str]] | None = None,
    ) -> None:
        """Initialise the OpenAI client from config values.

        Args:
            cfg: Teacher configuration.  `cfg.api_key` falls back to
                 ``NVIDIA_API_KEY`` (for NIM) then ``OPENAI_API_KEY``
                 environment variables when not set in the config.
            examples: Optional few-shot example pairs injected into every prompt.
        """
        super().__init__()
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("Install openai: uv add openai") from e

        self._cfg = cfg
        self._examples = examples or []
        if self._examples:
            print(f"Loaded {len(self._examples)} few-shot examples.")

        # Resolution order: config value → NVIDIA_API_KEY → OPENAI_API_KEY → "none"
        # NVIDIA_API_KEY is checked first so NIM users don't need to set OPENAI_API_KEY.
        api_key = (
            getattr(cfg, "api_key", None)
            or os.environ.get("NVIDIA_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "none"
        )
        base_url = getattr(cfg, "base_url", None)

        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
        print(
            f"OpenAI-compatible teacher: {cfg.model} @ {base_url or 'api.openai.com'}"
        )

    def _call(self, instruction: str, system_prompt: str, retries: int = 3) -> str:
        """Call the Chat Completions API with retry logic.

        Retries on ``RateLimitError`` with exponential back-off.
        Re-raises ``APIError`` on the final attempt.

        Args:
            instruction: The user instruction to send.
            system_prompt: System message for the conversation.
            retries: Maximum number of attempts before giving up.

        Returns:
            The model's response text, or an empty string if all retries fail.
        """
        from openai import RateLimitError, APIError

        messages = self._build_messages(instruction, system_prompt)

        for attempt in range(retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._cfg.model,
                    messages=messages,
                    temperature=self._cfg.temperature,
                    max_tokens=self._cfg.max_tokens,
                    extra_body=self._cfg.extra_body or {},
                )
                return response.choices[0].message.content or ""
            except RateLimitError:
                wait = 2**attempt
                print(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
            except APIError:
                if attempt == retries - 1:
                    raise
                time.sleep(1)

        print(
            f"Warning: all {retries} API attempts failed for instruction: {instruction[:60]!r} — returning empty string."
        )
        return ""

    def generate(self, instruction: str, system_prompt: str) -> str:
        """Generate a single response via the Chat Completions API.

        Args:
            instruction: The user instruction to respond to.
            system_prompt: System message prepended to the conversation.

        Returns:
            Generated text string.
        """
        return self._call(instruction, system_prompt)

    def generate_batch(self, instructions: list[str], system_prompt: str) -> list[str]:
        """Generate responses sequentially (no native batch API support).

        Args:
            instructions: List of instructions to process.
            system_prompt: Shared system message for all instructions.

        Returns:
            List of generated strings in the same order as `instructions`.
        """
        return [self.generate(inst, system_prompt) for inst in instructions]

    def close(self) -> None:
        """No-op: the HTTP client does not hold persistent connections."""
        pass
