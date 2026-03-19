"""
Multi-agent math workflow — evolvable with SkyDiscover.

SkyDiscover will ONLY mutate code inside:
  # EVOLVE-BLOCK-START
  ...
  # EVOLVE-BLOCK-END

Everything outside is stable scaffold for evaluation.
"""

import os, re, time, json
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, List
from openai import OpenAI

# ----------------------------
# Stable LLM wrapper (fixed)
# ----------------------------
_solver_model = os.getenv("SOLVER_MODEL", "gpt-4o-mini")
_is_reasoning = _solver_model.startswith(("o1", "o3", "o4", "gpt-5"))

_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


def call_llm(system_text: str, user_text: str) -> str:
    """Call the LLM with a system + user message. Returns the response text."""
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    kwargs = {
        "model": _solver_model,
        "messages": messages,
        "max_tokens": 2048 if _is_reasoning else 256,
    }
    if not _is_reasoning:
        kwargs["temperature"] = 0
    resp = _client.chat.completions.create(**kwargs)
    text = (resp.choices[0].message.content or "").strip() if resp.choices else ""
    return text


def extract_final_answer(text: str) -> str:
    """Extract the final numeric/symbolic answer from LLM output."""
    if not text:
        return "__NO_ANSWER__"

    m = re.search(r"FINAL ANSWER:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    tail = (m.group(1).strip() if m else text.strip())

    m = re.search(r"\\boxed\{([^}]*)\}", tail)
    if m:
        return m.group(1).strip()

    m = re.search(r"-?\d+(?:\.\d+)?", tail)
    if m and len(tail) <= 50:
        return m.group(0)

    return "__NO_ANSWER__"


# ----------------------------
# Stable dataclass definitions (fixed)
# ----------------------------
@dataclass
class AgentSpec:
    name: str
    kind: str  # "translator" | "reasoner" | "verifier" | "generic"
    system: str = ""


# ============================================================
# EVOLVE-BLOCK-START
# Everything inside this block is allowed to change.
# You can:
# - Edit agent system prompts
# - Add/delete agents by editing AGENT_SPECS
# - Rewire EDGES (list of (source, target) tuples)
# - Add coordination logic (retry rules, handoff conditions)
# - Tweak verification rules and thresholds
# - Add helper constants or small policy functions
# DO NOT change function signatures outside this block.
# ============================================================

AGENT_SPECS: List[AgentSpec] = [
    AgentSpec(
        name="translator",
        kind="translator",
        system=(
            "You are the translator agent. Rewrite the user's math problem "
            "into clean, precise mathematical notation. Do NOT solve the problem. "
            "Output only the rewritten problem statement."
        ),
    ),
    AgentSpec(
        name="reasoner",
        kind="reasoner",
        system=(
            "You are the reasoner agent. Solve the given math problem step by step. "
            "After your reasoning, output your answer on the last line as: "
            "FINAL ANSWER: <answer>"
        ),
    ),
    AgentSpec(
        name="verifier",
        kind="verifier",
        system="",
    ),
]

EDGES: List[Tuple[str, str]] = [
    ("START", "translator"),
    ("translator", "reasoner"),
    ("reasoner", "verifier"),
    ("verifier", "END"),
]

MAX_TURNS = 10

DISALLOWED_WORDS: List[str] = []

VERIFICATION_MAX_LENGTH = 500

def should_retry(verification_result: str) -> bool:
    """Return True if the verification failed and we should retry reasoning."""
    return False

# ============================================================
# EVOLVE-BLOCK-END
# ============================================================


# ----------------------------
# Stable graph execution (fixed)
# ----------------------------
def normalize_edges(
    agent_specs: List[AgentSpec], edges: List[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    """Validate and repair the agent graph edges."""
    names = [s.name for s in agent_specs]
    valid = set(names) | {"START", "END"}

    cleaned = [(u, v) for (u, v) in edges if u in valid and v in valid]
    if not cleaned:
        chain = ["START"] + names + ["END"]
        return list(zip(chain[:-1], chain[1:]))

    # Check reachability from START to END
    adj: Dict[str, List[str]] = {}
    for u, v in cleaned:
        adj.setdefault(u, []).append(v)

    seen: set = set()
    stack = ["START"]
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        for y in adj.get(x, []):
            stack.append(y)

    if "END" not in seen or any(n not in seen for n in names):
        chain = ["START"] + names + ["END"]
        return list(zip(chain[:-1], chain[1:]))

    return cleaned


def verify_answer(raw_text: str, extracted: str) -> Tuple[bool, str]:
    """Verify the extracted answer meets basic quality checks."""
    if extracted in ("__NO_ANSWER__", "", None):
        return False, "missing_answer"

    low = (raw_text or "").lower()
    if any(w in low for w in DISALLOWED_WORDS):
        return False, "contains_disallowed_words"

    if len((raw_text or "").strip()) > VERIFICATION_MAX_LENGTH:
        return False, "too_long"

    return True, "ok"


def _execute_agent(spec: AgentSpec, context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single agent node and return updated context."""
    if spec.kind == "translator":
        question = context["question"]
        translated = call_llm(spec.system, question).strip()
        if not translated:
            translated = question  # fallback to original
        context["translated"] = translated
        context["messages"].append(
            {"name": spec.name, "role": "translator", "content": translated}
        )

    elif spec.kind == "reasoner":
        problem = context.get("translated", context["question"])
        raw = call_llm(spec.system, problem).strip()
        answer = extract_final_answer(raw)
        context["raw_reasoning"] = raw
        context["extracted_answer"] = answer
        context["messages"].append(
            {"name": spec.name, "role": "reasoner", "content": raw}
        )

    elif spec.kind == "verifier":
        raw = context.get("raw_reasoning", "")
        extracted = context.get("extracted_answer", "__NO_ANSWER__")
        ok, reason = verify_answer(raw, extracted)
        verdict = "PASS" if ok else f"FAIL ({reason})"
        context["verified"] = ok
        context["verification_reason"] = reason
        context["messages"].append(
            {"name": spec.name, "role": "verifier", "content": f"VERIFICATION: {verdict}"}
        )

    elif spec.kind == "generic":
        # Generic agent: takes last message content, processes it
        last_content = context["messages"][-1]["content"] if context["messages"] else context["question"]
        system = spec.system or "You are a helpful agent. Process the input and improve it."
        result = call_llm(system, last_content).strip()
        if not result:
            result = last_content
        context["messages"].append(
            {"name": spec.name, "role": "generic", "content": result}
        )

    return context


def run_pipeline(question: str) -> Dict[str, Any]:
    """Execute the agent pipeline on a single question."""
    edges = normalize_edges(AGENT_SPECS, EDGES)

    # Build adjacency list
    adj: Dict[str, List[str]] = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)

    # Map agent names to specs
    spec_map = {s.name: s for s in AGENT_SPECS}

    context: Dict[str, Any] = {
        "question": question,
        "messages": [],
        "extracted_answer": "__NO_ANSWER__",
        "verified": False,
        "verification_reason": "not_run",
    }

    t0 = time.time()

    # Walk the graph from START
    current_nodes = adj.get("START", [])
    visited = set()
    steps = 0

    while current_nodes and steps < MAX_TURNS:
        next_nodes = []
        for node_name in current_nodes:
            if node_name == "END" or node_name in visited:
                continue
            visited.add(node_name)
            steps += 1

            if node_name in spec_map:
                context = _execute_agent(spec_map[node_name], context)

            # Check retry logic
            if node_name in spec_map and spec_map[node_name].kind == "verifier":
                if should_retry(context.get("verification_reason", "")):
                    # Re-queue the reasoner if retry is needed
                    for spec in AGENT_SPECS:
                        if spec.kind == "reasoner":
                            visited.discard(spec.name)
                            next_nodes.append(spec.name)
                            break
                    continue

            for target in adj.get(node_name, []):
                if target not in visited:
                    next_nodes.append(target)

        current_nodes = next_nodes

    dt = time.time() - t0

    return {
        "pred": context.get("extracted_answer", "__NO_ANSWER__"),
        "verified": context.get("verified", False),
        "latency_s": float(dt),
        "messages": context.get("messages", []),
        "num_agents": len(AGENT_SPECS),
    }


# Fixed entrypoint called by evaluator
def run_agentic(question: str) -> Dict[str, Any]:
    return run_pipeline(question)
