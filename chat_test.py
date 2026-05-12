import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import anthropic
from dotenv import load_dotenv
from openai import APITimeoutError as OpenAITimeoutError
from openai import APIError as OpenAIAPIError
from openai import OpenAI

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_THINKING_MODEL = "claude-sonnet-4-5"
DEFAULT_HISTORY_FILE = "chat_history.json"
DEFAULT_TIMEOUT_SECONDS = 45.0

EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit"}
MODEL_MODE_STANDARD = "standard"
MODEL_MODE_THINKING = "thinking"


@dataclass
class ModeSelection:
    mode: str
    model: str


def _env_to_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_to_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _safe_load_history(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _safe_save_history(path: Path, payload: Dict[str, Any]) -> None:
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"[warn] Failed to save history to '{path}': {exc}")


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
        return "\n".join(chunks).strip()
    return str(content)


class OpenAIChatSession:
    """Stateful OpenAI Chat Completions session with persistence."""

    def __init__(self, model: str, timeout_seconds: float, system_prompt: str) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=timeout_seconds,
        )
        self.messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]

    def load_messages(self, history_messages: List[Dict[str, Any]]) -> None:
        if history_messages:
            self.messages = history_messages

    def ask(self, user_text: str) -> tuple[str, Dict[str, Any]]:
        self.messages.append({"role": "user", "content": user_text})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=0.7,
        )
        assistant_text = response.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": assistant_text})
        usage = response.usage
        metrics = {
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
            "reasoning_tokens": None,
        }
        return assistant_text, metrics


class AnthropicThinkingSession:
    """Stateful Anthropic thinking session with reasoning metrics."""

    def __init__(
        self,
        model: str,
        timeout_seconds: float,
        system_prompt: str,
        thinking_budget_tokens: int,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.thinking_budget_tokens = thinking_budget_tokens
        self.client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            timeout=timeout_seconds,
        )
        self.messages: List[Dict[str, Any]] = []

    def load_messages(self, history_messages: List[Dict[str, Any]]) -> None:
        if history_messages:
            self.messages = history_messages

    def ask(self, user_text: str) -> tuple[str, Dict[str, Any]]:
        self.messages.append({"role": "user", "content": user_text})
        response = self.client.messages.create(
            model=self.model,
            system=self.system_prompt,
            messages=self.messages,
            thinking={"type": "enabled", "budget_tokens": self.thinking_budget_tokens},
            max_tokens=2048,
        )

        assistant_parts: List[str] = []
        thinking_blocks = 0
        thinking_chars = 0

        for block in response.content:
            block_type = getattr(block, "type", "")
            if block_type == "text":
                assistant_parts.append(getattr(block, "text", ""))
            if block_type == "thinking":
                thinking_blocks += 1
                thinking_chars += len(getattr(block, "thinking", ""))

        assistant_text = "\n".join(part for part in assistant_parts if part).strip()
        self.messages.append({"role": "assistant", "content": assistant_text})

        usage = getattr(response, "usage", None)
        metrics = {
            "input_tokens": getattr(usage, "input_tokens", None) if usage else None,
            "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
            "reasoning_tokens": None,
            "thinking_blocks": thinking_blocks,
            "thinking_chars": thinking_chars,
        }
        return assistant_text, metrics


def _load_session_history(
    history_file: Path,
    mode: str,
    model: str,
    system_prompt: str,
) -> List[Dict[str, Any]]:
    payload = _safe_load_history(history_file)
    if not payload:
        return []

    if payload.get("mode") != mode:
        return []

    history_messages = payload.get("messages", [])
    if not isinstance(history_messages, list):
        return []

    if mode == MODEL_MODE_STANDARD:
        if not history_messages:
            return []
        first = history_messages[0]
        if first.get("role") != "system":
            history_messages.insert(0, {"role": "system", "content": system_prompt})
    return history_messages


def _print_history_preview(messages: List[Dict[str, Any]], limit: int = 6) -> None:
    if not messages:
        print("[history] empty")
        return
    print(f"[history] showing last {min(limit, len(messages))} messages:")
    for msg in messages[-limit:]:
        role = str(msg.get("role", "unknown"))
        text = _extract_text_content(msg.get("content", ""))
        short = text.replace("\n", " ").strip()
        if len(short) > 120:
            short = short[:117] + "..."
        print(f"  - {role}: {short}")


def _save_session_history(
    history_file: Path,
    mode: str,
    model: str,
    messages: List[Dict[str, Any]],
) -> None:
    serializable: List[Dict[str, Any]] = []
    for msg in messages:
        serializable.append(
            {
                "role": str(msg.get("role", "")),
                "content": _extract_text_content(msg.get("content", "")),
            }
        )

    payload = {
        "mode": mode,
        "model": model,
        "messages": serializable,
    }
    _safe_save_history(history_file, payload)


def _choose_mode(default_mode: str) -> str:
    default_choice = "1" if default_mode == MODEL_MODE_STANDARD else "2"
    print("Choose chat mode:")
    print("1) Standard (OpenAI Chat Completions)")
    print("2) Thinking (Claude 4.5 Sonnet + reasoning metrics)")
    choice = input(f"Your choice [1/2] (Enter={default_choice}): ").strip().lower()
    if choice == "":
        choice = default_choice
    return MODEL_MODE_STANDARD if choice == "1" else MODEL_MODE_THINKING


def _print_metrics(mode: str, metrics: Dict[str, Any]) -> None:
    in_tokens = metrics.get("input_tokens")
    out_tokens = metrics.get("output_tokens")
    if mode == MODEL_MODE_THINKING:
        print(
            "[reasoning] "
            f"in={in_tokens}, out={out_tokens}, "
            f"thinking_blocks={metrics.get('thinking_blocks', 0)}, "
            f"thinking_chars={metrics.get('thinking_chars', 0)}"
        )
    else:
        print(f"[usage] in={in_tokens}, out={out_tokens}")


def run_demo() -> None:
    """Console assistant with persistent history and two model modes."""
    load_dotenv()

    timeout_seconds = _env_to_float("REQUEST_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
    system_prompt = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.")
    history_file = Path(os.getenv("CHAT_HISTORY_FILE", DEFAULT_HISTORY_FILE))
    openai_model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    anthropic_model = os.getenv("ANTHROPIC_THINKING_MODEL", DEFAULT_ANTHROPIC_THINKING_MODEL)
    default_mode = os.getenv("CHAT_MODEL_MODE", MODEL_MODE_THINKING).strip().lower()
    thinking_budget = int(os.getenv("ANTHROPIC_THINKING_BUDGET_TOKENS", "1024"))

    selected_mode = _choose_mode(default_mode)
    selection = ModeSelection(
        mode=selected_mode,
        model=openai_model if selected_mode == MODEL_MODE_STANDARD else anthropic_model,
    )

    print(
        f"[startup] mode={selection.mode}, model={selection.model}, "
        f"timeout={timeout_seconds}s, history_file={history_file}"
    )
    raw_history = _safe_load_history(history_file)
    if raw_history:
        saved_count = len(raw_history.get("messages", [])) if isinstance(raw_history.get("messages"), list) else 0
        print(
            f"[startup] found_history mode={raw_history.get('mode')}, "
            f"model={raw_history.get('model')}, messages={saved_count}"
        )

    if selection.mode == MODEL_MODE_STANDARD:
        session = OpenAIChatSession(
            model=selection.model,
            timeout_seconds=timeout_seconds,
            system_prompt=system_prompt,
        )
    else:
        session = AnthropicThinkingSession(
            model=selection.model,
            timeout_seconds=timeout_seconds,
            system_prompt=system_prompt,
            thinking_budget_tokens=thinking_budget,
        )

    loaded_history = _load_session_history(
        history_file=history_file,
        mode=selection.mode,
        model=selection.model,
        system_prompt=system_prompt,
    )
    session.load_messages(loaded_history)
    print(f"[startup] loaded_messages={len(loaded_history)}")
    if loaded_history:
        _print_history_preview(loaded_history)
    print("Chat started. Type 'exit' to stop.")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in EXIT_COMMANDS:
            print("Bye!")
            break
        if user_input.lower() == "/history":
            _print_history_preview(session.messages)
            continue
        if not user_input:
            continue

        try:
            answer, metrics = session.ask(user_input)
            print(f"Assistant: {answer}\n")
            _print_metrics(selection.mode, metrics)
            _save_session_history(
                history_file=history_file,
                mode=selection.mode,
                model=selection.model,
                messages=session.messages,
            )
        except (OpenAITimeoutError, anthropic.APITimeoutError):
            print("[error] Request timeout. Try again or increase REQUEST_TIMEOUT_SECONDS.")
        except (OpenAIAPIError, anthropic.APIError) as exc:
            print(f"[error] API request failed: {exc}")
        except Exception as exc:  # safety net for CLI continuity
            print(f"[error] Unexpected error: {exc}")


if __name__ == "__main__":
    run_demo()
