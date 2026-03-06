"""Abstract base class for all teacher backends."""

from abc import ABC, abstractmethod


class TeacherBackend(ABC):
    """Interface every teacher backend must implement.

    Few-shot examples are stored on the instance at construction time
    (via `_examples`) and injected into every prompt as user/assistant
    turns before the new instruction.

    Subclasses must implement `generate`, `generate_batch`, and `close`.
    """

    def __init__(self) -> None:
        """Initialise the instance-level examples list."""
        # Instance variable — not a class variable — to avoid shared mutable state
        self._examples: list[dict[str, str]] = []

    @abstractmethod
    def generate(self, instruction: str, system_prompt: str) -> str:
        """Generate one response for the given instruction.

        Args:
            instruction: The user-facing instruction to respond to.
            system_prompt: The system prompt to prepend to every conversation.

        Returns:
            The generated text, or an empty string on failure.
        """
        ...

    @abstractmethod
    def generate_batch(self, instructions: list[str], system_prompt: str) -> list[str]:
        """Generate responses for a batch of instructions (order-preserving).

        Args:
            instructions: List of instruction strings to process.
            system_prompt: Shared system prompt for all instructions.

        Returns:
            List of generated strings in the same order as `instructions`.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release any held resources (models, connections, etc.)."""
        ...

    def _build_messages(
        self, instruction: str, system_prompt: str
    ) -> list[dict[str, str]]:
        """Build a messages list with optional few-shot examples.

        Structure:
            [system]
            [user: example_1] [assistant: example_1_output]
            ...
            [user: new_instruction]

        Args:
            instruction: The new instruction to append at the end.
            system_prompt: Optional system message (skipped if empty).

        Returns:
            A list of role/content dicts suitable for a Chat Completions API.
        """
        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for ex in self._examples:
            messages.append({"role": "user", "content": ex["instruction"]})
            messages.append({"role": "assistant", "content": ex["output"]})

        messages.append({"role": "user", "content": instruction})
        return messages
