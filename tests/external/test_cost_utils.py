from types import SimpleNamespace

import pytest

from skydiscover.extras.external.cost_utils import (
    build_cost_event,
    make_llm_cost_summary,
)


def test_build_cost_event_uses_explicit_prices():
    usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=250, total_tokens=1250)

    event = build_cost_event(
        model_name="test-model",
        usage=usage,
        input_price_per_million_tokens=2.5,
        output_price_per_million_tokens=10.0,
    )

    assert event["input_tokens"] == 1000
    assert event["output_tokens"] == 250
    assert event["total_tokens"] == 1250
    assert event["input_cost_usd"] == pytest.approx(0.0025)
    assert event["output_cost_usd"] == pytest.approx(0.0025)
    assert event["total_cost_usd"] == pytest.approx(0.0050)


def test_make_llm_cost_summary_aggregates_events_and_extra_costs():
    summary = make_llm_cost_summary(
        [
            {
                "usage_category": "generation",
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "input_cost_usd": 0.1,
                "output_cost_usd": 0.2,
                "total_cost_usd": 0.3,
            },
            {
                "usage_category": "generation",
                "input_tokens": 50,
                "output_tokens": 10,
                "total_tokens": 60,
                "input_cost_usd": 0.05,
                "output_cost_usd": 0.1,
                "total_cost_usd": 0.15,
            },
        ],
        extra_category_costs={"embedding": 0.4},
    )

    assert summary["by_category"]["generation"]["call_count"] == 2
    assert summary["by_category"]["generation"]["input_tokens"] == 150
    assert summary["by_category"]["generation"]["output_tokens"] == 30
    assert summary["by_category"]["generation"]["total_cost_usd"] == pytest.approx(0.45)
    assert summary["by_category"]["embedding"]["total_cost_usd"] == pytest.approx(0.4)
    assert summary["total"]["call_count"] == 2
    assert summary["total"]["total_cost_usd"] == pytest.approx(0.85)
