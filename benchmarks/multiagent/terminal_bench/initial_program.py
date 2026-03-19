"""
Multi-agent terminal scaffold for SkyDiscover.

Only the LLM client, primitive tool implementations, and evaluator-facing
entrypoints are fixed. Nearly the entire scaffold architecture lives inside
the EVOLVE block so search can mutate the agent roster, routing, prompts,
memory, parsers, context management, and completion logic.
"""

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ═══════════════════════════════════════════════════════════════════════════
# Fixed infrastructure — do NOT modify
# ═══════════════════════════════════════════════════════════════════════════

_solver_model = os.getenv("SOLVER_MODEL", "gpt-4o-mini")
_is_reasoning = _solver_model.startswith(("o1", "o3", "o4", "gpt-5"))
_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


def call_llm_multi(system_text: str, messages: list, max_tokens: int = 4096) -> str:
    """Multi-turn LLM call. Returns the assistant's response text."""
    full = [{"role": "system", "content": system_text}] + messages
    kwargs = {"model": _solver_model, "messages": full}
    kwargs["max_tokens"] = max_tokens if not _is_reasoning else 8192
    if not _is_reasoning:
        kwargs["temperature"] = 0
    resp = _client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip() if resp.choices else ""


def run_shell(cmd: str, timeout: int = 30, cwd: Optional[str] = None) -> Dict[str, Any]:
    """Execute a shell command. Returns {stdout, stderr, returncode}."""
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


def list_dir(path: str = ".") -> List[str]:
    try:
        return sorted(os.listdir(path))
    except Exception as e:
        return [f"ERROR: {e}"]


_TOOL_REGISTRY = {
    "run_shell": run_shell,
    "read_file": read_file,
    "write_file": write_file,
    "list_dir": list_dir,
}


def _dispatch_tool(name: str, args: Dict[str, Any], work_dir: Optional[str], timeout: int) -> Any:
    fn = _TOOL_REGISTRY.get(name)
    if not fn:
        return f"Unknown tool: {name}"
    if name == "run_shell":
        args.setdefault("cwd", work_dir)
        args.setdefault("timeout", timeout)
    try:
        return fn(**args)
    except Exception as e:
        return f"Tool error: {e}"


# ═══════════════════════════════════════════════════════════════════════════
# EVOLVE-BLOCK-START
# Nearly the full scaffold is evolvable:
# - agent roster / count / roles
# - prompts and output schema
# - orchestration graph and handoff policy
# - memory, history pruning, summarization
# - parsing, retry, verification, completion logic
# - tool permissions and observation formatting
# ═══════════════════════════════════════════════════════════════════════════

MAX_GLOBAL_TURNS = 18
COMMAND_TIMEOUT = 30
MAX_CONTEXT_MESSAGES = 60
MAX_OBSERVATION_CHARS = 5000
MAX_MEMORY_CHARS = 6000
START_AGENT = "planner"


@dataclass
class AgentSpec:
    name: str
    role: str
    goal: str
    allowed_tools: List[str]


@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]


@dataclass
class ParsedResponse:
    analysis: str = ""
    memory_update: str = ""
    next_agent: str = ""
    task_complete: bool = False
    commands: List[ToolCall] = field(default_factory=list)
    error: str = ""


AGENT_SPECS: List[AgentSpec] = [
    AgentSpec(
        name="planner",
        role="planner",
        goal=(
            "Break the task into executable steps, decide what must be inspected "
            "before acting, and route work to the best specialist."
        ),
        allowed_tools=["list_dir", "read_file", "run_shell"],
    ),
    AgentSpec(
        name="researcher",
        role="researcher",
        goal=(
            "Inspect the workspace, gather evidence from commands and files, and "
            "surface constraints or failure causes before execution."
        ),
        allowed_tools=["list_dir", "read_file", "run_shell"],
    ),
    AgentSpec(
        name="executor",
        role="executor",
        goal=(
            "Make the required filesystem or shell changes. Prefer targeted actions "
            "over large speculative command batches."
        ),
        allowed_tools=["run_shell", "write_file", "read_file", "list_dir"],
    ),
    AgentSpec(
        name="verifier",
        role="verifier",
        goal=(
            "Check the end state rigorously. Re-run targeted commands or file reads, "
            "and only signal completion if the task is truly done."
        ),
        allowed_tools=["run_shell", "read_file", "list_dir"],
    ),
]


ALLOWED_HANDOFFS: Dict[str, List[str]] = {
    "planner": ["researcher", "executor", "verifier"],
    "researcher": ["planner", "executor", "verifier"],
    "executor": ["planner", "researcher", "verifier"],
    "verifier": ["planner", "executor"],
}


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
    if len(rendered) <= MAX_MEMORY_CHARS:
        return rendered
    return truncate_output(rendered, MAX_MEMORY_CHARS)


def manage_context(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if len(messages) <= MAX_CONTEXT_MESSAGES:
        return messages

    keep_start = 2
    keep_end = max(6, MAX_CONTEXT_MESSAGES - keep_start - 1)
    pruned_count = max(0, len(messages) - keep_start - keep_end)
    summary = {
        "role": "user",
        "content": (
            f"[context summary] {pruned_count} earlier messages were pruned.\n"
            "Rely on the shared memory and the recent transcript."
        ),
    }
    return messages[:keep_start] + [summary] + messages[-keep_end:]


def build_roster_summary() -> str:
    lines = []
    for spec in AGENT_SPECS:
        tools = ", ".join(spec.allowed_tools)
        lines.append(f"- {spec.name}: {spec.goal} Allowed tools: {tools}.")
    return "\n".join(lines)


def build_system_prompt(spec: AgentSpec, memory: Dict[str, Any], work_dir: Optional[str]) -> str:
    handoffs = ", ".join(ALLOWED_HANDOFFS.get(spec.name, [])) or spec.name
    tool_lines = "\n".join(f"- {tool}" for tool in spec.allowed_tools)
    return f"""\
You are the {spec.name} agent in a multi-agent terminal scaffold.

Role goal:
{spec.goal}

Agent roster:
{build_roster_summary()}

Workspace root:
{work_dir or "."}

Allowed handoffs from your role:
{handoffs}

Shared memory:
{compact_memory(memory)}

Available tools for your role:
{tool_lines}

Rules:
1. Use absolute paths in commands and file operations.
2. Do not claim completion unless the task is actually done.
3. If another agent should act next, name it in <next_agent>.
4. If you run tools, keep them focused and relevant to your role.
5. The verifier should be the final authority on completion.
6. If a command fails, explain why and choose a different next step.

Respond in this exact structure:
<analysis>
What you learned and what matters now.
</analysis>
<memory_update>
Facts or decisions worth keeping in shared memory.
</memory_update>
<next_agent>planner|researcher|executor|verifier</next_agent>
<commands>
TOOL: run_shell | {{"cmd": "..."}}  # optional
TOOL: read_file | {{"path": "..."}} # optional
</commands>
<task_complete>true</task_complete>  # only if the task is complete
"""


def parse_response(response: str) -> ParsedResponse:
    parsed = ParsedResponse()

    def _extract(tag: str) -> str:
        match = re.search(rf"<{tag}>(.*?)</{tag}>", response, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    parsed.analysis = _extract("analysis")
    parsed.memory_update = _extract("memory_update")
    parsed.next_agent = _extract("next_agent").lower().strip()
    parsed.task_complete = bool(
        re.search(r"<task_complete>\s*true\s*</task_complete>", response, re.IGNORECASE)
    )

    commands_text = _extract("commands") or response
    for line in commands_text.splitlines():
        stripped = line.strip()
        if not stripped.upper().startswith("TOOL:"):
            continue
        rest = stripped[5:].strip()
        if "|" not in rest:
            continue
        name_part, _, args_part = rest.partition("|")
        name = name_part.strip().lower().replace("-", "_")
        args_str = args_part.strip()
        try:
            args = json.loads(args_str)
            if not isinstance(args, dict):
                args = {"cmd": str(args)} if name == "run_shell" else {}
        except json.JSONDecodeError:
            if name == "run_shell":
                args = {"cmd": args_str.strip("\"' ")}
            elif name in ("read_file", "list_dir"):
                args = {"path": args_str.strip("\"' ")}
            else:
                continue
        if name in _TOOL_REGISTRY:
            parsed.commands.append(ToolCall(name=name, args=args))

    if not parsed.analysis:
        parsed.error = "missing_analysis"
    return parsed


def filter_tool_calls(agent: AgentSpec, calls: List[ToolCall]) -> List[ToolCall]:
    allowed = set(agent.allowed_tools)
    return [call for call in calls if call.name in allowed]


def format_observation(agent_name: str, call: ToolCall, result: Any) -> str:
    prefix = f"[agent={agent_name} tool={call.name}]"
    if call.name == "run_shell" and isinstance(result, dict):
        rc = result.get("returncode", -1)
        out = truncate_output(result.get("stdout", "").strip(), 3500)
        err = truncate_output(result.get("stderr", "").strip(), 1500)
        parts = [prefix, f"$ {call.args.get('cmd', '?')}", f"returncode={rc}"]
        if out:
            parts.append(f"stdout:\n{out}")
        if err:
            parts.append(f"stderr:\n{err}")
        return "\n".join(parts)
    if call.name == "read_file":
        return f"{prefix}\n{truncate_output(str(result), 3500)}"
    if call.name == "write_file":
        return f"{prefix}\nwrite_ok={bool(result)} path={call.args.get('path', '?')}"
    if call.name == "list_dir":
        rendered = "\n".join(result[:80]) if isinstance(result, list) else str(result)
        return f"{prefix}\n{truncate_output(rendered, 3000)}"
    return f"{prefix}\n{truncate_output(str(result), 3000)}"


def update_memory(
    memory: Dict[str, Any],
    agent_name: str,
    parsed: ParsedResponse,
    observations: List[str],
    had_failure: bool,
) -> None:
    memory["last_agent"] = agent_name
    memory["had_recent_failure"] = had_failure
    if parsed.memory_update:
        memory.setdefault("notes", []).append(f"{agent_name}: {parsed.memory_update}")
    if parsed.analysis:
        memory["last_analysis"] = parsed.analysis
    if observations:
        memory["last_observation"] = truncate_output("\n\n".join(observations), 2500)
    notes = memory.get("notes", [])
    if len(notes) > 12:
        memory["notes"] = notes[-12:]


def choose_next_agent(agent_name: str, parsed: ParsedResponse, had_failure: bool) -> str:
    requested = parsed.next_agent
    allowed = set(ALLOWED_HANDOFFS.get(agent_name, []))
    if requested and requested in allowed:
        return requested
    if parsed.task_complete and agent_name != "verifier":
        return "verifier"
    if agent_name == "planner":
        return "researcher"
    if agent_name == "researcher":
        return "executor" if not had_failure else "planner"
    if agent_name == "executor":
        return "verifier" if not had_failure else "planner"
    return "planner"


def parse_error_message(agent_name: str, parsed: ParsedResponse) -> str:
    if parsed.error == "missing_analysis":
        return (
            f"The {agent_name} response was malformed. Include <analysis>, "
            "<memory_update>, <next_agent>, and <commands>."
        )
    return (
        f"The {agent_name} response had no actionable tool calls or valid handoff. "
        "Either delegate clearly or execute a concrete verification/action step."
    )


def run_multiagent_scaffold(task: str, work_dir: Optional[str]) -> Dict[str, Any]:
    t0 = time.time()
    if work_dir:
        os.makedirs(work_dir, exist_ok=True)

    spec_map = {spec.name: spec for spec in AGENT_SPECS}
    current_agent = START_AGENT if START_AGENT in spec_map else AGENT_SPECS[0].name
    messages: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": (
                f"TASK:\n{task}\n\n"
                f"Workspace directory: {work_dir or '.'}\n"
                "Coordinate across agents to solve the task and verify completion."
            ),
        }
    ]
    memory: Dict[str, Any] = {
        "task": task,
        "workspace": work_dir or ".",
        "notes": ["Task received. Planner should decompose before execution."],
        "had_recent_failure": False,
    }

    total_tool_calls = 0
    success_count = 0
    turns_used = 0
    done_signal = False
    tool_log: List[Dict[str, Any]] = []

    for turn in range(MAX_GLOBAL_TURNS):
        turns_used = turn + 1
        messages = manage_context(messages)
        agent = spec_map[current_agent]
        system_prompt = build_system_prompt(agent, memory, work_dir)
        response = call_llm_multi(system_prompt, messages)
        messages.append({"role": "assistant", "content": f"[{agent.name}]\n{response}"})
        parsed = parse_response(response)

        allowed_calls = filter_tool_calls(agent, parsed.commands)
        if not allowed_calls and not parsed.task_complete and not parsed.next_agent:
            messages.append({"role": "user", "content": parse_error_message(agent.name, parsed)})
            current_agent = choose_next_agent(agent.name, parsed, had_failure=True)
            memory["had_recent_failure"] = True
            continue

        observations: List[str] = []
        had_failure = False
        for call in allowed_calls:
            result = _dispatch_tool(call.name, dict(call.args), work_dir, COMMAND_TIMEOUT)
            observations.append(format_observation(agent.name, call, result))
            total_tool_calls += 1

            ok = False
            if call.name == "run_shell" and isinstance(result, dict):
                ok = result.get("returncode") == 0
                if ok:
                    success_count += 1
                else:
                    had_failure = True
            else:
                ok = bool(result)
                if not ok:
                    had_failure = True

            tool_log.append(
                {
                    "agent": agent.name,
                    "tool": call.name,
                    "args": str(call.args)[:200],
                    "ok": ok,
                }
            )

        update_memory(memory, agent.name, parsed, observations, had_failure)

        if observations:
            observation_blob = "\n\n".join(observations)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Observations from {agent.name}:\n{observation_blob}\n\n"
                        f"Shared memory now:\n{compact_memory(memory)}"
                    ),
                }
            )

        if parsed.task_complete:
            if agent.name == "verifier" and not had_failure:
                done_signal = True
                break
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"{agent.name} believes the task may be done. "
                        "The verifier must confirm with explicit checks before finishing."
                    ),
                }
            )
            current_agent = "verifier"
            continue

        current_agent = choose_next_agent(agent.name, parsed, had_failure)

    return {
        "total_steps": total_tool_calls,
        "success_count": success_count,
        "latency_s": time.time() - t0,
        "num_agents": len(AGENT_SPECS),
        "done_signal": done_signal,
        "turns_used": turns_used,
        "tool_call_log": tool_log[:60],
    }


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
        "num_agents": int(result.get("num_agents", len(AGENT_SPECS))),
        "done_signal": bool(result.get("done_signal", False)),
        "turns_used": int(result.get("turns_used", 0)),
        "tool_call_log": list(result.get("tool_call_log", []))[:60],
    }


def run_agentic(task: str, work_dir: Optional[str] = None) -> Dict[str, Any]:
    """Entrypoint called by the evaluator."""
    return run_pipeline(task, work_dir=work_dir)
