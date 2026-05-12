import os
from typing import List, Dict

import anthropic
from openai import OpenAI
from dotenv import load_dotenv

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_ANTHROPIC_STANDARD_MODEL = "claude-haiku-4-5"


def _env_to_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_verbose_explanation(default: bool = False) -> bool:
    if os.getenv("CHAT_VERBOSE_EXPLANATION") is not None:
        return _env_to_bool("CHAT_VERBOSE_EXPLANATION", default=default)
    return _env_to_bool("OPENAI_VERBOSE_EXPLANATION", default=default)


class ChatSession:
    """Simple stateful chat session using Chat Completions API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        system_prompt: str = "You are a helpful assistant.",
        verbose_explanation: bool | None = None,
    ) -> None:
        load_dotenv()
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.verbose_explanation = (
            verbose_explanation
            if verbose_explanation is not None
            else _resolve_verbose_explanation(default=False)
        )
        self.messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    def ask(self, user_text: str) -> str:
        """
        Sends user message to Chat Completions API and stores full dialog context.
        Returns assistant response text.
        """
        final_user_text = user_text
        if self.verbose_explanation:
            final_user_text = (
                f"{user_text}\n\n"
                "Please add a short step-by-step explanation of your answer."
            )

        self.messages.append({"role": "user", "content": final_user_text})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=0.7,
        )

        assistant_text = response.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": assistant_text})
        return assistant_text

    def reset(self) -> None:
        """Resets conversation while keeping system prompt."""
        system_message = self.messages[0]
        self.messages = [system_message]


class AnthropicChatSession:
    """Stateful chat session for Anthropic Messages API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        system_prompt: str = "You are a helpful assistant.",
        verbose_explanation: bool | None = None,
        thinking_budget_tokens: int | None = None,
        enable_thinking: bool = True,
        fallback_model: str = DEFAULT_ANTHROPIC_MODEL,
    ) -> None:
        load_dotenv()
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model = model or os.getenv("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
        self.system_prompt = system_prompt
        self.enable_thinking = enable_thinking
        self.fallback_model = fallback_model
        self.verbose_explanation = (
            verbose_explanation
            if verbose_explanation is not None
            else _resolve_verbose_explanation(default=False)
        )
        env_budget = os.getenv("ANTHROPIC_THINKING_BUDGET_TOKENS")
        self.thinking_budget_tokens = thinking_budget_tokens or int(env_budget or "1024")
        self.messages: List[Dict[str, str]] = []

    def _create_message(self, model_name: str):
        request_kwargs = {
            "model": model_name,
            "system": self.system_prompt,
            "messages": self.messages,
            "max_tokens": 2048,
        }
        if self.enable_thinking:
            request_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget_tokens,
            }
        return self.client.messages.create(
            **request_kwargs,
        )

    def ask(self, user_text: str) -> str:
        """Sends user message to Anthropic API and stores full dialog context."""
        final_user_text = user_text
        if self.verbose_explanation:
            final_user_text = (
                f"{user_text}\n\n"
                "Please add a short step-by-step explanation of your answer."
            )

        self.messages.append({"role": "user", "content": final_user_text})
        try:
            response = self._create_message(self.model)
        except anthropic.NotFoundError:
            if self.model != self.fallback_model:
                print(
                    f"Warning: model '{self.model}' was not found. "
                    f"Falling back to '{self.fallback_model}'."
                )
                self.model = self.fallback_model
                response = self._create_message(self.model)
            else:
                raise RuntimeError(
                    f"Anthropic model '{self.model}' was not found. "
                    "Set ANTHROPIC_MODEL in .env to a valid model id."
                ) from None

        assistant_text_parts = []
        for block in response.content:
            if getattr(block, "type", "") == "text":
                assistant_text_parts.append(block.text)
        assistant_text = "\n".join(assistant_text_parts).strip()

        self.messages.append({"role": "assistant", "content": assistant_text})
        return assistant_text

    def reset(self) -> None:
        """Resets conversation while keeping system prompt."""
        self.messages = []


def run_demo() -> None:
    """Tiny CLI demo to keep chatting until user exits."""
    load_dotenv()
    provider = os.getenv("CHAT_PROVIDER", "anthropic").strip().lower()

    if provider == "openai":
        chat = ChatSession()
        active_model = chat.model
        model_mode = "openai"
    else:
        thinking_model = os.getenv("ANTHROPIC_THINKING_MODEL", DEFAULT_ANTHROPIC_MODEL)
        standard_model = os.getenv(
            "ANTHROPIC_STANDARD_MODEL",
            DEFAULT_ANTHROPIC_STANDARD_MODEL,
        )
        default_mode = os.getenv("CHAT_MODEL_MODE", "thinking").strip().lower()
        default_choice = "1" if default_mode == "thinking" else "2"

        print("Choose Anthropic model mode:")
        print(f"1) Thinking model ({thinking_model})")
        print(f"2) Standard model ({standard_model})")
        choice = input(f"Your choice [1/2] (Enter={default_choice}): ").strip().lower()

        use_thinking = choice not in {"2", "standard", "s"}
        selected_model = thinking_model if use_thinking else standard_model
        model_mode = "thinking" if use_thinking else "standard"
        fallback_model = thinking_model if use_thinking else standard_model
        chat = AnthropicChatSession(
            model=selected_model,
            enable_thinking=use_thinking,
            fallback_model=fallback_model,
        )
        active_model = chat.model

    mode = "ON" if chat.verbose_explanation else "OFF"
    print(
        f"Chat started. Provider: {provider}. Mode: {model_mode}. Model: {active_model}. "
        f"Type 'exit' to stop. Verbose explanation mode: {mode}"
    )

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Bye!")
            break
        if not user_input:
            continue

        answer = chat.ask(user_input)
        print(f"Assistant: {answer}\n")


if __name__ == "__main__":
    run_demo()
