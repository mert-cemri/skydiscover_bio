"""Cost helpers shared by external backend wrappers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from skydiscover.llm.cost import extract_usage_counts


def get_model_token_prices(
    model_name: str,
    input_price_per_million_tokens: Optional[float] = None,
    output_price_per_million_tokens: Optional[float] = None,
) -> tuple[float, float]:
    """Resolve per-token prices from explicit config or LiteLLM pricing metadata."""
    if input_price_per_million_tokens is not None and output_price_per_million_tokens is not None:
        return (
            float(input_price_per_million_tokens) / 1_000_000.0,
            float(output_price_per_million_tokens) / 1_000_000.0,
        )

    try:
        import litellm

        info = litellm.get_model_info(model_name)
    except Exception:
        info = {}

    input_cost_per_token = float(info.get("input_cost_per_token") or 0.0)
    output_cost_per_token = float(info.get("output_cost_per_token") or 0.0)

    if input_price_per_million_tokens is not None:
        input_cost_per_token = float(input_price_per_million_tokens) / 1_000_000.0
    if output_price_per_million_tokens is not None:
        output_cost_per_token = float(output_price_per_million_tokens) / 1_000_000.0

    return input_cost_per_token, output_cost_per_token


def build_cost_event(
    *,
    model_name: str,
    usage: Any,
    usage_category: str = "generation",
    input_price_per_million_tokens: Optional[float] = None,
    output_price_per_million_tokens: Optional[float] = None,
) -> Dict[str, Any]:
    """Build one JSON-serializable usage event from a provider response."""
    input_tokens, output_tokens, total_tokens = extract_usage_counts(usage)
    input_cost_per_token, output_cost_per_token = get_model_token_prices(
        model_name,
        input_price_per_million_tokens=input_price_per_million_tokens,
        output_price_per_million_tokens=output_price_per_million_tokens,
    )
    input_cost_usd = input_tokens * input_cost_per_token
    output_cost_usd = output_tokens * output_cost_per_token
    return {
        "usage_category": usage_category,
        "model_name": model_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_cost_usd": input_cost_usd,
        "output_cost_usd": output_cost_usd,
        "total_cost_usd": input_cost_usd + output_cost_usd,
    }


def append_cost_event(path: str, event: Dict[str, Any]) -> None:
    """Append one cost event as JSONL."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def load_cost_events(path: str) -> List[Dict[str, Any]]:
    """Load cost events from a JSONL file."""
    p = Path(path)
    if not p.exists():
        return []
    events: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def make_llm_cost_summary(
    events: Iterable[Dict[str, Any]],
    extra_category_costs: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Convert raw cost events into the standard summary structure."""
    by_category: Dict[str, Dict[str, Any]] = {}

    def _ensure_bucket(name: str) -> Dict[str, Any]:
        return by_category.setdefault(
            name,
            {
                "call_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "input_cost_usd": 0.0,
                "output_cost_usd": 0.0,
                "total_cost_usd": 0.0,
            },
        )

    for event in events:
        bucket = _ensure_bucket(event.get("usage_category") or "generation")
        bucket["call_count"] += 1
        bucket["input_tokens"] += int(event.get("input_tokens") or 0)
        bucket["output_tokens"] += int(event.get("output_tokens") or 0)
        bucket["total_tokens"] += int(event.get("total_tokens") or 0)
        bucket["input_cost_usd"] += float(event.get("input_cost_usd") or 0.0)
        bucket["output_cost_usd"] += float(event.get("output_cost_usd") or 0.0)
        bucket["total_cost_usd"] += float(event.get("total_cost_usd") or 0.0)

    for category, total_cost in (extra_category_costs or {}).items():
        bucket = _ensure_bucket(category)
        bucket["total_cost_usd"] += float(total_cost or 0.0)

    total = {
        "call_count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "input_cost_usd": 0.0,
        "output_cost_usd": 0.0,
        "total_cost_usd": 0.0,
    }
    for bucket in by_category.values():
        total["call_count"] += bucket["call_count"]
        total["input_tokens"] += bucket["input_tokens"]
        total["output_tokens"] += bucket["output_tokens"]
        total["total_tokens"] += bucket["total_tokens"]
        total["input_cost_usd"] += bucket["input_cost_usd"]
        total["output_cost_usd"] += bucket["output_cost_usd"]
        total["total_cost_usd"] += bucket["total_cost_usd"]

    return {"total": total, "by_category": by_category}
