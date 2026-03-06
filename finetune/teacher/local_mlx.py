"""Local MLX teacher using mlx_lm.generate."""

from __future__ import annotations

from finetune.teacher.base import TeacherBackend
from finetune.config import TeacherConfig


class LocalMLXTeacher(TeacherBackend):
    """Runs a local MLX model as the teacher backend.

    Loads the model and tokenizer from Hugging Face (via mlx_lm.load) at
    construction time. Uses `batch_generate` for throughput when available,
    falling back to sequential `generate` otherwise.
    """

    def __init__(self, cfg: TeacherConfig, examples: list[dict[str, str]] | None = None) -> None:
        """Load the MLX model and tokenizer specified in the config.

        Args:
            cfg: Teacher configuration (model path, sampling parameters, etc.).
            examples: Optional few-shot example pairs to inject into every prompt.
        """
        super().__init__()
        from mlx_lm import load

        self._cfg = cfg
        self._examples = examples or []
        if self._examples:
            print(f"Loaded {len(self._examples)} few-shot examples.")
        print(f"Loading teacher model: {cfg.model}")
        self._model, self._tokenizer = load(cfg.model)

    def _to_prompt(self, messages: list[dict[str, str]]) -> str:
        """Apply the tokenizer's chat template to produce a single prompt string.

        Args:
            messages: List of role/content dicts (system, user, assistant).

        Returns:
            Formatted prompt string ready for mlx_lm.generate.
        """
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(self, instruction: str, system_prompt: str) -> str:
        """Generate a single response for the given instruction.

        Args:
            instruction: The user instruction to respond to.
            system_prompt: System message prepended to the conversation.

        Returns:
            Generated text string.
        """
        from mlx_lm import generate

        messages = self._build_messages(instruction, system_prompt)
        prompt = self._to_prompt(messages)
        return generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self._cfg.max_tokens,
            temp=self._cfg.temperature,
            verbose=False,
        )

    def generate_batch(self, instructions: list[str], system_prompt: str) -> list[str]:
        """Use mlx_lm batch_generate for throughput on Apple Silicon.

        Falls back to sequential generation if batch_generate is unavailable,
        printing a warning so the slowdown is visible.

        Args:
            instructions: List of instructions to generate responses for.
            system_prompt: Shared system message for all instructions.

        Returns:
            List of generated strings in the same order as `instructions`.
        """
        prompts = [
            self._to_prompt(self._build_messages(inst, system_prompt))
            for inst in instructions
        ]
        try:
            from mlx_lm import batch_generate

            result = batch_generate(
                self._model,
                self._tokenizer,
                prompts=prompts,
                max_tokens=self._cfg.max_tokens,
                temp=self._cfg.temperature,
            )
            return result.texts if hasattr(result, "texts") else list(result)
        except (ImportError, AttributeError):
            print("Warning: batch_generate unavailable — falling back to sequential generation.")
            return [self.generate(inst, system_prompt) for inst in instructions]

    def close(self) -> None:
        """Delete the model and tokenizer to free MLX memory."""
        if hasattr(self, "_model"):
            del self._model
        if hasattr(self, "_tokenizer"):
            del self._tokenizer
