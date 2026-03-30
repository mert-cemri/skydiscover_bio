"""
Stronger Harbor-native terminal scaffold seed for SkyDiscover.

This version keeps the benchmark contract intact while upgrading the runtime
and action interface substantially: native tool calling, an interactive shell
session, richer tools, explicit verification state, and a completion gate.
"""

import json
import os
import pty
import re
import select
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from urllib import error as urlerror
from urllib import request as urlrequest

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - exercised only in minimal task images
    OpenAI = None

# ═══════════════════════════════════════════════════════════════════════════
# Fixed infrastructure — do NOT modify
# ═══════════════════════════════════════════════════════════════════════════

_solver_model = os.getenv("SOLVER_MODEL", "gpt-4o-mini")
_is_reasoning = _solver_model.startswith(("o1", "o3", "o4", "gpt-5"))
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _dict_to_obj(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _dict_to_obj(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_dict_to_obj(item) for item in value]
    return value


class _FallbackChatCompletions:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def create(self, **kwargs: Any) -> Any:
        payload = json.dumps(kwargs).encode("utf-8")
        req = urlrequest.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlrequest.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
        except urlerror.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI HTTP {exc.code}: {body}") from exc
        return _dict_to_obj(data)


class _FallbackOpenAI:
    def __init__(self, api_key: str) -> None:
        self.chat = SimpleNamespace(completions=_FallbackChatCompletions(api_key))


_api_key = os.environ.get("OPENAI_API_KEY", "")
_client = OpenAI(api_key=_api_key) if OpenAI is not None else _FallbackOpenAI(api_key=_api_key)


def call_llm_multi(system_text: str, messages: list, max_tokens: int = 4096) -> str:
    """Multi-turn LLM call. Returns the assistant's response text."""
    full = [{"role": "system", "content": system_text}] + messages
    kwargs = {
        "model": _solver_model,
        "messages": full,
        "max_tokens": max_tokens if max_tokens > 0 else 4096,
    }
    if not _is_reasoning:
        kwargs["temperature"] = 0
    resp = _client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip() if resp.choices else ""


def call_llm_tools(
    system_text: str,
    messages: list,
    tools: list,
    max_tokens: int = 4096,
) -> Dict[str, Any]:
    """Tool-calling LLM call. Returns {content, tool_calls}."""
    full = [{"role": "system", "content": system_text}] + messages
    kwargs = {
        "model": _solver_model,
        "messages": full,
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": max_tokens if max_tokens > 0 else 4096,
    }
    if not _is_reasoning:
        kwargs["temperature"] = 0
    try:
        resp = _client.chat.completions.create(**kwargs)
    except Exception as _llm_exc:
        import sys as _sys
        print(f"SKYDISCOVER_LLM_ERROR: model={kwargs.get('model')} msgs={len(kwargs.get('messages',[]))} max_tokens={kwargs.get('max_tokens')} temperature={kwargs.get('temperature','N/A')} error={repr(_llm_exc)}", file=_sys.stderr, flush=True)
        raise
    if not resp.choices:
        return {"content": "", "tool_calls": []}
    message = resp.choices[0].message
    tool_calls = []
    if getattr(message, "tool_calls", None):
        for tc in message.tool_calls:
            tool_calls.append(
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "{}",
                    },
                }
            )
    return {"content": message.content or "", "tool_calls": tool_calls}


class InteractiveShellSession:
    """Lightweight in-container interactive shell backed by a PTY."""

    MARKER_PREFIX = "__SKYDISCOVER_CMD_DONE__"

    def __init__(self, work_dir: Optional[str]) -> None:
        self.work_dir = work_dir or "."
        master_fd, slave_fd = pty.openpty()
        env = os.environ.copy()
        env.setdefault("TERM", "xterm-256color")
        env.setdefault("DEBIAN_FRONTEND", "noninteractive")
        env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
        env.setdefault("PIP_ROOT_USER_ACTION", "ignore")
        env.setdefault("CI", "1")
        env.setdefault("PYTHONUNBUFFERED", "1")
        self.proc = subprocess.Popen(
            ["/bin/bash", "--noprofile", "--norc"],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            cwd=self.work_dir,
            env=env,
            close_fds=True,
        )
        os.close(slave_fd)
        self.master_fd = master_fd
        os.set_blocking(self.master_fd, False)
        self.transcript = ""
        self._drain(0.2)
        self.run_command("export PS1=''", timeout_s=0.2)
        self.run_command("stty -echo 2>/dev/null || true", timeout_s=0.2)

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                self.proc.wait(timeout=1)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        try:
            os.close(self.master_fd)
        except Exception:
            pass

    def _read_available(self, timeout_s: float) -> str:
        chunks: List[str] = []
        end = time.time() + max(0.0, timeout_s)
        while True:
            remaining = max(0.0, end - time.time())
            ready, _, _ = select.select([self.master_fd], [], [], remaining)
            if not ready:
                break
            try:
                raw = os.read(self.master_fd, 4096)
            except BlockingIOError:
                break
            if not raw:
                break
            chunks.append(raw.decode("utf-8", errors="replace"))
            if time.time() >= end:
                break
        return "".join(chunks)

    def _drain(self, timeout_s: float = 0.1) -> str:
        text = self._read_available(timeout_s)
        if text:
            self.transcript += text
        return text

    def _clean_output(self, text: str, command: str) -> str:
        text = _ANSI_RE.sub("", text).replace("\r", "")
        lines = text.splitlines()
        while lines and lines[0].strip() in ("", command.strip()):
            lines.pop(0)
        return "\n".join(lines).strip()

    def run_command(self, cmd: str, timeout_s: float) -> Dict[str, Any]:
        timeout_s = max(0.1, min(float(timeout_s or 0.1), float(MAX_COMMAND_TIMEOUT)))
        marker = f"{self.MARKER_PREFIX}{uuid.uuid4().hex}"
        payload = f"{cmd}\nprintf '\\n{marker}:%s\\n' \"$?\"\n"
        try:
            os.write(self.master_fd, payload.encode("utf-8"))
        except Exception as exc:
            return {"stdout": "", "stderr": str(exc), "returncode": -1}

        collected = ""
        deadline = time.time() + timeout_s
        marker_text = f"{marker}:"

        while time.time() < deadline:
            chunk = self._read_available(0.2)
            if chunk:
                collected += chunk
                self.transcript += chunk
                if marker_text in collected:
                    break

        if marker_text not in collected:
            try:
                os.write(self.master_fd, b"\x03")
            except Exception:
                pass
            self._drain(0.2)
            return {
                "stdout": self._clean_output(collected, cmd),
                "stderr": f"Timed out after {timeout_s}s",
                "returncode": -1,
            }

        stdout_text, marker_suffix = collected.split(marker_text, 1)
        rc_match = re.search(r"(-?\d+)", marker_suffix)
        returncode = int(rc_match.group(1)) if rc_match else 0
        return {
            "stdout": self._clean_output(stdout_text, cmd),
            "stderr": "",
            "returncode": returncode,
        }

    def wait(self, seconds: float) -> Dict[str, Any]:
        waited = max(0.1, min(float(seconds or 0.1), 10.0))
        time.sleep(waited)
        return {"stdout": self.capture_recent_output(), "stderr": "", "returncode": 0}

    def capture_recent_output(self, max_chars: int = 4000) -> str:
        text = _ANSI_RE.sub("", self.transcript).replace("\r", "")
        return text[-max_chars:]


def run_shell(cmd: str, timeout: int = 30, cwd: Optional[str] = None) -> Dict[str, Any]:
    """One-shot shell command. Retained for compatibility and quick probes."""
    try:
        r = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return {"stdout": r.stdout, "stderr": r.stderr, "returncode": r.returncode}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"Timed out after {timeout}s", "returncode": -1}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "returncode": -1}


def read_file(path: str, max_bytes: int = 50000) -> str:
    try:
        with open(path, "r", errors="replace") as f:
            return f.read(max_bytes)
    except Exception as e:
        return f"ERROR: {e}"


def write_file(path: str, content: str) -> bool:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return True
    except Exception:
        return False


def edit_file(path: str, old_text: str, new_text: str) -> bool:
    try:
        with open(path, "r", errors="replace") as f:
            text = f.read()
        if old_text not in text:
            return False
        with open(path, "w") as f:
            f.write(text.replace(old_text, new_text, 1))
        return True
    except Exception:
        return False


def list_dir(path: str = ".") -> List[str]:
    try:
        return sorted(os.listdir(path))
    except Exception as e:
        return [f"ERROR: {e}"]


def search_files(query: str, path: str = ".", max_matches: int = 20, max_bytes: int = 200000) -> str:
    matches: List[str] = []
    lowered = query.lower()
    visited_bytes = 0
    try:
        for root, _, files in os.walk(path):
            for filename in files:
                if len(matches) >= max_matches:
                    break
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, path)
                if lowered in filename.lower() or lowered in rel_path.lower():
                    matches.append(f"{rel_path}: filename-match")
                    continue
                try:
                    with open(file_path, "r", errors="ignore") as f:
                        text = f.read(4000)
                        visited_bytes += len(text)
                    if lowered in text.lower():
                        matches.append(f"{rel_path}: content-match")
                    if visited_bytes >= max_bytes:
                        break
                except Exception:
                    continue
            if len(matches) >= max_matches or visited_bytes >= max_bytes:
                break
    except Exception as exc:
        return f"ERROR: {exc}"
    return "\n".join(matches) if matches else "No matches found."


# ═══════════════════════════════════════════════════════════════════════════
# EVOLVE-BLOCK-START
# Strong seed architecture is fully evolvable:
# - tool schemas and prompts
# - verification policy
# - memory layout
# - retry and completion logic
# - seed profile (single agent vs two phase)
# ═══════════════════════════════════════════════════════════════════════════

MAX_GLOBAL_TURNS = int(os.getenv("TBENCH_MAX_GLOBAL_TURNS", "24"))
MAX_ACTIONS_PER_TURN = int(os.getenv("TBENCH_MAX_ACTIONS_PER_TURN", "3"))
COMMAND_TIMEOUT = int(os.getenv("TBENCH_COMMAND_TIMEOUT", "25"))
MAX_COMMAND_TIMEOUT = 180
MAX_RETRY_HEAVY_COMMANDS = int(os.getenv("TBENCH_MAX_RETRY_HEAVY_COMMANDS", "2"))
MAX_CONTEXT_MESSAGES = 40
MAX_MEMORY_CHARS = 7000
MAX_OBSERVATION_CHARS = 5000
MAX_TOOL_CALL_TOKENS = 4096
SEED_PROFILE = os.getenv("TBENCH_SEED_PROFILE", "single_agent_native").strip().lower()
SEED_NUM_AGENTS = 2 if SEED_PROFILE == "two_phase" else 1
ALLOW_IMAGE_READ = False

ACCEPTANCE_PLAYBOOK = [
    "bash /app/tests/test.sh",
    "bash /tests/verify.sh",
    "python3 -m pytest /tests/test_outputs.py -x -q",
    "pytest /tests/test_outputs.py -x -q",
    "python -m pytest /tests/test_outputs.py -x -q",
    "make test",
    "npm test --silent",
    "cargo test",
    "go test ./...",
]

MUTATION_TOKENS = ("git ", "mv ", "cp ", "rm ", "sed ", "perl ", "tee ", " > ", "mkdir ", "touch ")
VERIFY_TOKENS = (
    "pytest",
    "make test",
    "npm test",
    "cargo test",
    "go test",
    "ctest",
    "verify.sh",
    "test -f",
    "test -d",
    "cmp ",
    "diff ",
    "grep -q",
)
# ACCEPTANCE_TOKENS is stricter than VERIFY_TOKENS: only real test runners count as
# acceptance evidence. Broad shell checks (test -f, grep -q, diff, cmp) can fire
# trivially and cause false-positive done_signal on hard tasks.
ACCEPTANCE_TOKENS = (
    "pytest",
    "make test",
    "npm test",
    "cargo test",
    "go test",
    "ctest",
    "verify.sh",
    "test.sh",
)
HEAVY_COMMAND_TOKENS = (
    "cmake",
    "make",
    "ninja",
    "pytest",
    "mvn",
    "gradle",
    "cargo",
    "go test",
    "npm test",
    "pip install",
    "uv pip",
    "docker build",
    "apt-get",
    "apt ",
)


@dataclass
class ToolCall:
    id: str
    name: str
    args: Dict[str, Any]


@dataclass
class LLMDecision:
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    task_complete: bool = False


TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "execute_commands",
            "description": "Execute 1-3 focused shell commands in the interactive terminal. Include analysis, plan, and intent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "analysis": {"type": "string"},
                    "plan": {"type": "string"},
                    "intent": {"type": "string", "enum": ["inspect", "modify", "verify"]},
                    "commands": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "cmd": {"type": "string"},
                                "timeout_s": {"type": "number"},
                            },
                            "required": ["cmd"],
                        },
                    },
                },
                "required": ["analysis", "plan", "intent", "commands"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the filesystem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_bytes": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a full text file to disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace one exact text occurrence in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List directory contents.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search filenames and nearby file contents for a string query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string"},
                    "max_matches": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait_seconds",
            "description": "Wait briefly and inspect any new terminal output.",
            "parameters": {
                "type": "object",
                "properties": {"seconds": {"type": "number"}},
                "required": ["seconds"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_complete",
            "description": "Request task completion. The first call asks for confirmation; the second succeeds only after a successful verification step since the last mutation.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def truncate_output(text: str, max_bytes: int = MAX_OBSERVATION_CHARS) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    portion = max_bytes // 2
    first = encoded[:portion].decode("utf-8", errors="ignore")
    last = encoded[-portion:].decode("utf-8", errors="ignore")
    omitted = len(encoded) - len(first.encode("utf-8")) - len(last.encode("utf-8"))
    return f"{first}\n[... {omitted} bytes omitted ...]\n{last}"


def compact_memory(memory: Dict[str, Any]) -> str:
    rendered = json.dumps(memory, ensure_ascii=True, indent=2, sort_keys=True)
    return truncate_output(rendered, MAX_MEMORY_CHARS)


def manage_context(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(messages) <= MAX_CONTEXT_MESSAGES:
        return messages
    keep_start = 2
    keep_end = max(8, MAX_CONTEXT_MESSAGES - keep_start - 1)
    tail = messages[-keep_end:]
    # Ensure the tail does not start mid-turn: if the first message is a 'tool'
    # result, its preceding assistant+tool_calls message was truncated away, which
    # the API rejects.  Walk forward until we hit a clean turn boundary (a non-tool
    # message that is not an orphaned tool result).
    while tail and tail[0].get("role") == "tool":
        tail = tail[1:]
    if not tail:
        tail = messages[-2:]
    summary = {
        "role": "user",
        "content": "Earlier turns were pruned for context. Use the shared memory summary and the latest observations.",
    }
    return messages[:keep_start] + [summary] + tail


def build_system_prompt(memory: Dict[str, Any], work_dir: Optional[str]) -> str:
    role_line = (
        "You are a single, high-agency terminal agent with strong verification discipline."
        if SEED_PROFILE == "single_agent_native"
        else "You are acting as a compact planner-verifier harness: inspect, modify, then verify before finishing."
    )
    return f"""\
{role_line}

You are running inside a Harbor-evaluated container task.
Workspace root: {work_dir or "."}

Critical rules:
1. Use the provided tools only.
2. Keep commands focused. Prefer 1-2 commands per batch and never exceed {MAX_ACTIONS_PER_TURN} commands in one `execute_commands` call.
3. Use `intent="verify"` only for concrete acceptance checks.
4. Never run `/tests/test.sh` from inside the scaffold. In this benchmark that script re-runs the scaffold recursively.
5. Before calling `task_complete`, identify the minimum files/state that must have changed and ensure you did not create unrelated side effects.
6. Completion requires a successful verify step and a successful acceptance command run since the last mutation.
7. Acceptance command must be a real test command (pytest/make test/npm test/etc), not ls/cat/grep.
8. Cache a working acceptance command when you discover one.
9. Current seed profile: {SEED_PROFILE}. Honor the current phase and avoid phase-skipping.

Preferred verification playbook:
- {" | ".join(ACCEPTANCE_PLAYBOOK)}

Shared memory:
{compact_memory(memory)}
"""


def _normalize_tool_calls(response: Dict[str, Any]) -> LLMDecision:
    decision = LLMDecision(content=response.get("content", ""))
    for tc in response.get("tool_calls", []):
        name = tc.get("function", {}).get("name", "")
        args_str = tc.get("function", {}).get("arguments", "{}")
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else dict(args_str or {})
        except Exception:
            args = {}
        decision.tool_calls.append(
            ToolCall(
                id=tc.get("id") or f"call_{len(decision.tool_calls)}",
                name=name,
                args=args,
            )
        )
        if name == "task_complete":
            decision.task_complete = True
    return decision


def _looks_mutating_command(cmd: str) -> bool:
    lowered = f" {cmd.lower()} "
    return any(token in lowered for token in MUTATION_TOKENS)


def _looks_verification_command(cmd: str) -> bool:
    lowered = cmd.lower()
    return any(token in lowered for token in VERIFY_TOKENS)


def _looks_acceptance_command(cmd: str) -> bool:
    lowered = cmd.lower()
    return any(token in lowered for token in ACCEPTANCE_TOKENS)


def _normalize_acceptance_command(cmd: str) -> str:
    lowered = " ".join(str(cmd).strip().lower().split())
    if lowered.startswith("python3 -m pytest "):
        lowered = lowered.replace("python3 -m pytest ", "pytest ", 1)
    elif lowered.startswith("python -m pytest "):
        lowered = lowered.replace("python -m pytest ", "pytest ", 1)
    elif lowered.startswith("uv run pytest "):
        lowered = lowered.replace("uv run pytest ", "pytest ", 1)
    return lowered


def _discover_acceptance_hint(session: InteractiveShellSession) -> str:
    probe = session.run_command(
        "if [ -f /app/tests/test.sh ]; then echo 'bash /app/tests/test.sh'; "
        "elif [ -f /tests/verify.sh ]; then echo 'bash /tests/verify.sh'; "
        "elif [ -f /tests/test_outputs.py ]; then echo 'python3 -m pytest /tests/test_outputs.py -x -q'; "
        "elif [ -f /tests/test.py ]; then echo 'python3 -m pytest /tests/test.py -x -q'; "
        "elif [ -f /app/package.json ]; then echo 'npm test --silent'; "
        "elif [ -f /app/Cargo.toml ]; then echo 'cargo test'; "
        "elif [ -f /app/go.mod ]; then echo 'go test ./...'; "
        "elif [ -f /app/Makefile ]; then echo 'make test'; "
        "else echo ''; fi",
        timeout_s=2.0,
    )
    if probe.get("returncode", -1) != 0:
        return ""
    return str(probe.get("stdout", "")).strip()


def _is_heavy_command(cmd: str) -> bool:
    lowered = cmd.lower()
    return any(token in lowered for token in HEAVY_COMMAND_TOKENS)


def _adaptive_timeout(cmd: str, requested_timeout_s: float, intent: str) -> float:
    timeout_s = max(1.0, min(float(requested_timeout_s or COMMAND_TIMEOUT), MAX_COMMAND_TIMEOUT))
    if _is_heavy_command(cmd):
        timeout_s = max(timeout_s, 90.0 if intent in ("modify", "verify") else 60.0)
    return min(timeout_s, MAX_COMMAND_TIMEOUT)


def _is_transient_failure(result: Dict[str, Any]) -> bool:
    stderr = str(result.get("stderr", "")).lower()
    stdout = str(result.get("stdout", "")).lower()
    haystack = f"{stdout}\n{stderr}"
    transient_tokens = (
        "timed out",
        "temporarily unavailable",
        "resource busy",
        "connection reset",
        "connection refused",
        "failed to fetch",
        "temporary failure",
        "network",
        "http 5",
        "rate limit",
        "try again",
        "lock",
    )
    return any(token in haystack for token in transient_tokens)


def _command_fingerprint(cmd: str) -> str:
    return " ".join(str(cmd).strip().lower().split())


def _execute_with_retry(session: InteractiveShellSession, cmd: str, timeout_s: float, intent: str) -> Dict[str, Any]:
    effective_timeout = _adaptive_timeout(cmd, timeout_s, intent)
    result = session.run_command(cmd, timeout_s=effective_timeout)
    retries = 0
    max_retries = MAX_RETRY_HEAVY_COMMANDS if _is_heavy_command(cmd) else 0
    while retries < max_retries and result.get("returncode", -1) != 0 and _is_transient_failure(result):
        retries += 1
        effective_timeout = min(MAX_COMMAND_TIMEOUT, effective_timeout * 1.5)
        result = session.run_command(cmd, timeout_s=effective_timeout)
    result["retries"] = retries
    result["effective_timeout_s"] = round(effective_timeout, 2)
    return result


def _split_compound_command(cmd: str) -> List[str]:
    if "&&" not in cmd:
        return [cmd]
    parts = [part.strip() for part in cmd.split("&&") if part.strip()]
    if len(parts) <= 1 or len(parts) > 3:
        return [cmd]
    if not any(_is_heavy_command(part) for part in parts):
        return [cmd]
    return parts


def _format_tool_result(name: str, result: Any) -> str:
    if isinstance(result, dict):
        rc = result.get("returncode", 0)
        out = truncate_output(result.get("stdout", "").strip(), 3000)
        err = truncate_output(result.get("stderr", "").strip(), 1200)
        parts = [f"returncode={rc}"]
        if out:
            parts.append(f"stdout:\n{out}")
        if err:
            parts.append(f"stderr:\n{err}")
        return "\n".join(parts)
    if isinstance(result, list):
        return truncate_output("\n".join(result), 3000)
    return truncate_output(str(result), 3000)


def _tool_result_message(tool_call_id: str, text: str) -> Dict[str, Any]:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": text}


def _run_initial_probe(session: InteractiveShellSession) -> str:
    probe = session.run_command(
        "pwd && echo '---' && ls -la && echo '---APP---' && (ls /app 2>/dev/null | sed -n '1,40p' || true)",
        timeout_s=2.5,
    )
    return _format_tool_result("initial_probe", probe)


def run_multiagent_scaffold(task: str, work_dir: Optional[str]) -> Dict[str, Any]:
    t0 = time.time()
    workspace = work_dir or "."
    os.makedirs(workspace, exist_ok=True)

    session = InteractiveShellSession(workspace)
    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                f"TASK:\n{task}\n\n"
                f"Workspace directory: {workspace}\n"
                "Solve the task with concrete evidence and careful verification."
            ),
        }
    ]
    memory: Dict[str, Any] = {
        "task": task,
        "workspace": workspace,
        "seed_profile": SEED_PROFILE,
        "acceptance_cmd": "",
        "notes": [],
        "last_analysis": "",
        "last_plan": "",
        "last_terminal_snapshot": "",
        "pending_completion": False,
        "mutation_id": 0,
        "verifier_success_since_mutation": False,
        "acceptance_verified_since_mutation": False,
        "last_verifier_cmd": "",
        "last_verifier_ok": False,
        "failure_counts": {},
        "phase": "plan" if SEED_PROFILE == "two_phase" else "direct",
    }

    initial_probe = _run_initial_probe(session)
    memory["last_terminal_snapshot"] = initial_probe
    acceptance_hint = _discover_acceptance_hint(session)
    if acceptance_hint:
        memory["acceptance_cmd"] = acceptance_hint
        memory.setdefault("notes", []).append(f"acceptance_hint: {acceptance_hint}")
    messages.append({"role": "user", "content": f"Initial probe:\n{initial_probe}"})

    total_tool_calls = 0
    success_count = 0
    turns_used = 0
    done_signal = False
    tool_log: List[Dict[str, Any]] = []

    try:
        for turn in range(MAX_GLOBAL_TURNS):
            turns_used = turn + 1
            messages = manage_context(messages)
            response = call_llm_tools(
                build_system_prompt(memory, workspace),
                messages,
                TOOL_DEFS,
                max_tokens=MAX_TOOL_CALL_TOKENS,
            )
            decision = _normalize_tool_calls(response)

            assistant_message: Dict[str, Any] = {
                "role": "assistant",
                "content": response.get("content", "") or "",
            }
            if decision.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.args),
                        },
                    }
                    for call in decision.tool_calls
                ]
            messages.append(assistant_message)

            if not decision.tool_calls:
                messages.append(
                    {
                        "role": "user",
                        "content": "No tool calls were returned. Use tools to inspect, modify, verify, or complete the task.",
                    }
                )
                continue

            action_count = 0
            turn_had_failure = False
            turn_had_verification_success = False
            observed_completion_request = False

            for call in decision.tool_calls:
                if call.name != "task_complete" and action_count >= MAX_ACTIONS_PER_TURN:
                    messages.append(
                        _tool_result_message(
                            call.id,
                            f"Skipped: exceeded MAX_ACTIONS_PER_TURN={MAX_ACTIONS_PER_TURN}.",
                        )
                    )
                    continue

                result_text = ""
                ok = False

                if call.name == "execute_commands":
                    action_count += 1
                    total_tool_calls += 1
                    analysis = str(call.args.get("analysis", "")).strip()
                    plan = str(call.args.get("plan", "")).strip()
                    intent = str(call.args.get("intent", "inspect")).strip().lower()
                    commands = call.args.get("commands", [])
                    memory["last_analysis"] = analysis
                    memory["last_plan"] = plan
                    if analysis:
                        memory.setdefault("notes", []).append(f"analysis: {analysis[:240]}")
                    if plan:
                        memory.setdefault("notes", []).append(f"plan: {plan[:240]}")
                    if SEED_PROFILE == "two_phase":
                        memory["phase"] = {
                            "inspect": "modify",
                            "modify": "verify",
                            "verify": "plan",
                        }.get(intent, memory.get("phase", "plan"))

                    command_results: List[str] = []
                    for item in commands[:MAX_ACTIONS_PER_TURN]:
                        cmd = str(item.get("cmd", "")).strip()
                        timeout_s = float(item.get("timeout_s", COMMAND_TIMEOUT))
                        if not cmd:
                            continue
                        for sub_cmd in _split_compound_command(cmd):
                            shell_result = _execute_with_retry(session, sub_cmd, timeout_s=timeout_s, intent=intent)
                            command_results.append(f"$ {sub_cmd}\n{_format_tool_result(call.name, shell_result)}")
                            cmd_ok = shell_result.get("returncode", -1) == 0
                            is_verify_attempt = intent == "verify" or _looks_verification_command(sub_cmd)
                            is_acceptance_attempt = _looks_acceptance_command(sub_cmd)
                            fingerprint = _command_fingerprint(sub_cmd)
                            ok = ok or cmd_ok
                            if cmd_ok:
                                success_count += 1
                                memory.setdefault("failure_counts", {}).pop(fingerprint, None)
                                if is_verify_attempt:
                                    turn_had_verification_success = True
                                    memory["last_verifier_cmd"] = sub_cmd
                                    memory["last_verifier_ok"] = True
                                    if not memory.get("acceptance_cmd") and is_acceptance_attempt:
                                        memory["acceptance_cmd"] = sub_cmd
                                    if (
                                        memory.get("acceptance_cmd")
                                        and is_acceptance_attempt
                                        and _normalize_acceptance_command(sub_cmd)
                                        == _normalize_acceptance_command(str(memory["acceptance_cmd"]))
                                    ):
                                        memory["acceptance_verified_since_mutation"] = True
                            else:
                                turn_had_failure = True
                                failure_counts = memory.setdefault("failure_counts", {})
                                failure_counts[fingerprint] = int(failure_counts.get(fingerprint, 0)) + 1
                                if is_verify_attempt:
                                    memory["last_verifier_cmd"] = sub_cmd
                                    memory["last_verifier_ok"] = False
                                if failure_counts[fingerprint] >= 2:
                                    memory.setdefault("notes", []).append(
                                        f"Repeated failure on `{sub_cmd[:120]}`; change strategy instead of retrying unchanged."
                                    )
                            if _looks_mutating_command(sub_cmd):
                                memory["mutation_id"] = int(memory.get("mutation_id", 0)) + 1
                                memory["verifier_success_since_mutation"] = False
                                memory["acceptance_verified_since_mutation"] = False
                            tool_log.append(
                                {
                                    "agent": "native",
                                    "tool": call.name,
                                    "args": sub_cmd[:200],
                                    "ok": cmd_ok,
                                }
                            )
                    result_text = "\n\n".join(command_results) or "No commands executed."
                    memory["pending_completion"] = False

                elif call.name == "read_file":
                    action_count += 1
                    total_tool_calls += 1
                    content = read_file(
                        str(call.args.get("path", "")),
                        max_bytes=int(call.args.get("max_bytes", 50000)),
                    )
                    ok = not str(content).startswith("ERROR:")
                    result_text = _format_tool_result(call.name, content)
                    if ok:
                        success_count += 1
                    else:
                        turn_had_failure = True
                    tool_log.append(
                        {
                            "agent": "native",
                            "tool": call.name,
                            "args": str(call.args)[:200],
                            "ok": ok,
                        }
                    )
                    memory["pending_completion"] = False

                elif call.name == "write_file":
                    action_count += 1
                    total_tool_calls += 1
                    ok = write_file(str(call.args.get("path", "")), str(call.args.get("content", "")))
                    result_text = f"write_ok={ok}"
                    if ok:
                        success_count += 1
                        memory["mutation_id"] = int(memory.get("mutation_id", 0)) + 1
                        memory["verifier_success_since_mutation"] = False
                        memory["acceptance_verified_since_mutation"] = False
                    else:
                        turn_had_failure = True
                    tool_log.append(
                        {
                            "agent": "native",
                            "tool": call.name,
                            "args": str(call.args)[:200],
                            "ok": ok,
                        }
                    )
                    memory["pending_completion"] = False

                elif call.name == "edit_file":
                    action_count += 1
                    total_tool_calls += 1
                    ok = edit_file(
                        str(call.args.get("path", "")),
                        str(call.args.get("old_text", "")),
                        str(call.args.get("new_text", "")),
                    )
                    result_text = f"edit_ok={ok}"
                    if ok:
                        success_count += 1
                        memory["mutation_id"] = int(memory.get("mutation_id", 0)) + 1
                        memory["verifier_success_since_mutation"] = False
                        memory["acceptance_verified_since_mutation"] = False
                    else:
                        turn_had_failure = True
                    tool_log.append(
                        {
                            "agent": "native",
                            "tool": call.name,
                            "args": str(call.args)[:200],
                            "ok": ok,
                        }
                    )
                    memory["pending_completion"] = False

                elif call.name == "list_dir":
                    action_count += 1
                    total_tool_calls += 1
                    listing = list_dir(str(call.args.get("path", ".")))
                    ok = not (listing and str(listing[0]).startswith("ERROR:"))
                    result_text = _format_tool_result(call.name, listing)
                    if ok:
                        success_count += 1
                    else:
                        turn_had_failure = True
                    tool_log.append(
                        {
                            "agent": "native",
                            "tool": call.name,
                            "args": str(call.args)[:200],
                            "ok": ok,
                        }
                    )
                    memory["pending_completion"] = False

                elif call.name == "search_files":
                    action_count += 1
                    total_tool_calls += 1
                    search_result = search_files(
                        str(call.args.get("query", "")),
                        path=str(call.args.get("path", ".")),
                        max_matches=int(call.args.get("max_matches", 20)),
                    )
                    ok = not search_result.startswith("ERROR:")
                    result_text = _format_tool_result(call.name, search_result)
                    if ok:
                        success_count += 1
                    else:
                        turn_had_failure = True
                    tool_log.append(
                        {
                            "agent": "native",
                            "tool": call.name,
                            "args": str(call.args)[:200],
                            "ok": ok,
                        }
                    )
                    memory["pending_completion"] = False

                elif call.name == "wait_seconds":
                    action_count += 1
                    total_tool_calls += 1
                    wait_result = session.wait(float(call.args.get("seconds", 1.0)))
                    ok = wait_result.get("returncode", -1) == 0
                    result_text = _format_tool_result(call.name, wait_result)
                    if ok:
                        success_count += 1
                    else:
                        turn_had_failure = True
                    tool_log.append(
                        {
                            "agent": "native",
                            "tool": call.name,
                            "args": str(call.args)[:200],
                            "ok": ok,
                        }
                    )
                    memory["pending_completion"] = False

                elif call.name == "task_complete":
                    observed_completion_request = True
                    completion_prompt = (
                        "Completion confirmation requested.\n"
                        f"Task: {task}\n\n"
                        "Before finishing, verify the minimum set of files/state changes required by the task.\n"
                        "Confirm there are no unrelated side effects.\n"
                        f"Known acceptance command: {memory.get('acceptance_cmd') or 'not discovered'}\n"
                        f"Verifier success since last mutation: {memory.get('verifier_success_since_mutation')}\n"
                        f"Acceptance command success since last mutation: {memory.get('acceptance_verified_since_mutation')}\n"
                        f"Last verifier command: {memory.get('last_verifier_cmd') or 'none'}\n"
                        "If you are truly done, call task_complete again after the verification evidence is sufficient."
                    )
                    if (
                        memory.get("pending_completion")
                        and memory.get("verifier_success_since_mutation")
                        and memory.get("acceptance_verified_since_mutation")
                    ):
                        done_signal = True
                        messages.append(_tool_result_message(call.id, "Task completion confirmed."))
                        break
                    memory["pending_completion"] = True
                    messages.append(_tool_result_message(call.id, completion_prompt))
                    continue
                else:
                    messages.append(
                        _tool_result_message(
                            call.id,
                            f"Unknown tool '{call.name}'. Use the provided tool definitions only.",
                        )
                    )
                    turn_had_failure = True
                    continue

                messages.append(_tool_result_message(call.id, result_text))
                memory["last_terminal_snapshot"] = session.capture_recent_output()

            if turn_had_verification_success:
                memory["verifier_success_since_mutation"] = True
                if SEED_PROFILE == "two_phase":
                    memory["phase"] = "plan"

            if done_signal:
                break

            if not observed_completion_request:
                memory["pending_completion"] = False

            if turn_had_failure:
                memory.setdefault("notes", []).append("A recent tool action failed; re-plan with tighter scope.")
            if len(memory.get("notes", [])) > 12:
                memory["notes"] = memory["notes"][-12:]

            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Latest terminal snapshot:\n{truncate_output(memory['last_terminal_snapshot'], 3000)}\n\n"
                        f"Shared memory now:\n{compact_memory(memory)}"
                    ),
                }
            )

        return {
            "total_steps": total_tool_calls,
            "success_count": success_count,
            "latency_s": time.time() - t0,
            "num_agents": SEED_NUM_AGENTS,
            "done_signal": done_signal,
            "turns_used": turns_used,
            "tool_call_log": tool_log[:80],
        }
    finally:
        session.close()


# ═══════════════════════════════════════════════════════════════════════════
# EVOLVE-BLOCK-END
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# Fixed entrypoints — do NOT modify
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(task: str, work_dir: Optional[str] = None) -> Dict[str, Any]:
    result = run_multiagent_scaffold(task, work_dir=work_dir)
    return {
        "total_steps": int(result.get("total_steps", 0)),
        "success_count": int(result.get("success_count", 0)),
        "latency_s": float(result.get("latency_s", 0.0)),
        "num_agents": int(result.get("num_agents", SEED_NUM_AGENTS)),
        "done_signal": bool(result.get("done_signal", False)),
        "turns_used": int(result.get("turns_used", 0)),
        "tool_call_log": list(result.get("tool_call_log", []))[:80],
    }


def run_agentic(task: str, work_dir: Optional[str] = None) -> Dict[str, Any]:
    """Entrypoint called by the evaluator."""
    return run_pipeline(task, work_dir=work_dir)
