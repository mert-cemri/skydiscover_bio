"""
AI-in-the-loop feedback reader for autonomous discovery steering.

Periodically analyses discovery statistics via an LLM and injects
strategic guidance into the system prompt -- the same injection point
used by HumanFeedbackReader, but fully automated.

Threading model:
    - ``read()``, ``apply_feedback()``, and ``maybe_trigger()`` are called
      on the **main** async thread.
    - ``_analyze_background()`` runs in a **background** ThreadPoolExecutor thread.
    - All shared mutable state is protected by ``self._lock``.
"""

import logging
import threading
import time as _time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Rough cost estimates per 1K tokens (input, output) for budget tracking.
# Conservative estimates -- overestimates are safer than underestimates.
_COST_PER_1K_INPUT = 0.003  # $3/M input tokens
_COST_PER_1K_OUTPUT = 0.015  # $15/M output tokens
_AVG_CHARS_PER_TOKEN = 3  # Conservative (overestimates token count = safer budget tracking)

_MAX_FEEDBACK_LOG = 100  # Max audit log entries to keep

_ANALYSIS_SYSTEM_PROMPT = """\
You are an AI advisor steering an evolutionary program discovery process.
You receive statistics about the current search state. Your job is to write
2-3 sentences of strategic guidance that will be appended to the code generation
prompt for the next iterations.

Guidelines:
- If scores are improving steadily, respond with exactly: "No changes needed."
- If stagnation is detected (many iterations without improvement), suggest a
  specific new algorithmic direction to explore.
- If parent reuse ratio is high (>0.5), suggest increasing diversity: mention
  specific alternative approaches (different data structures, algorithms, or
  mathematical techniques).
- Be specific and actionable. Name concrete algorithms, data structures, or
  optimisation strategies.
- Keep your response under 150 words.
- Do NOT repeat previous guidance verbatim.
- Treat scores as ground truth. Do NOT judge code quality by structure alone.
- Focus on WHAT to try, not HOW to code it.
"""


def _is_no_changes(text: str) -> bool:
    """Check if the response means 'no changes needed' (tolerant matching)."""
    return text.lower().rstrip(".!").strip() == "no changes needed"


class AIFeedbackReader:
    """Autonomous AI feedback that analyses discovery state and generates guidance.

    This class mirrors the HumanFeedbackReader interface (``read()``,
    ``apply_feedback()``) so it can be used alongside or instead of
    human feedback in the discovery controller.
    """

    def __init__(
        self,
        config: Any,  # AIFeedbackConfig
        call_llm_fn: Callable[..., str],
        problem_description: str = "",
    ):
        self._config = config
        self._call_llm = call_llm_fn
        self._problem_description = problem_description
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ai_feedback")
        self._lock = threading.Lock()

        # Feedback state (rolling buffer -- latest analysis only)
        self._active_feedback: str = ""
        self._pending_feedback: Optional[str] = None
        self._latest_artifact_summary: str = ""

        # Guardrail state
        self._last_analysis_iter: int = -999
        self._analysis_in_progress: bool = False
        self._cost_spent: float = 0.0
        self._budget_exhausted: bool = False

        # Auto-clear tracking
        self._iterations_with_feedback: int = 0
        self._best_score_at_feedback_start: float = float("-inf")
        self._previous_feedback: str = ""  # for dedup

        # Audit log
        self._feedback_log: List[dict] = []

        logger.info(
            "AIFeedbackReader initialized (model=%s, interval=%d, budget=$%.2f)",
            config.model,
            config.interval,
            config.max_cost_dollars,
        )

    # ------------------------------------------------------------------
    # Public interface (called on MAIN thread)
    # ------------------------------------------------------------------

    def read(self) -> str:
        """Return current AI feedback text. Promotes pending feedback if available."""
        with self._lock:
            if self._pending_feedback is not None:
                old = self._active_feedback
                self._active_feedback = self._pending_feedback
                self._pending_feedback = None
                if self._active_feedback != old:
                    if self._active_feedback:
                        logger.info(
                            "AI feedback activated (%d chars)", len(self._active_feedback)
                        )
                    else:
                        logger.info("AI feedback cleared")
            return self._active_feedback

    def apply_feedback(self, prompt: dict) -> dict:
        """Append AI guidance to the system prompt. Never raises."""
        try:
            feedback = self.read()
            if not feedback or _is_no_changes(feedback):
                return prompt
            prompt["system"] = (
                prompt["system"]
                + "\n\n## AI Analysis (suggestions — defer to Human Guidance above if present)\n"
                + feedback
            )
        except Exception:
            logger.debug("AI feedback apply failed", exc_info=True)
        return prompt

    def maybe_trigger(
        self,
        stats: dict,
        iteration: int,
        best_score: float,
    ) -> None:
        """Check guardrails and dispatch background analysis if appropriate.

        Called on the main thread from the monitor callback.
        ``stats`` must be a pre-computed snapshot (safe to pass to background).
        """
        with self._lock:
            if not self._should_analyze(iteration):
                return

            # Track iterations with active feedback for auto-clear
            if self._active_feedback:
                self._iterations_with_feedback += 1

                # Auto-clear: feedback has been active too long without improvement
                if self._iterations_with_feedback >= self._config.auto_clear_after:
                    if best_score <= self._best_score_at_feedback_start:
                        logger.info(
                            "AI feedback auto-cleared (no improvement after %d iterations)",
                            self._iterations_with_feedback,
                        )
                        self._active_feedback = ""
                        self._iterations_with_feedback = 0
                        self._last_analysis_iter = iteration  # cooldown
                        self._feedback_log.append({
                            "iteration": iteration,
                            "timestamp": _time.time(),
                            "action": "auto_cleared",
                        })
                        return
                    else:
                        # Feedback is helping -- reset counter
                        self._iterations_with_feedback = 0
                        self._best_score_at_feedback_start = best_score

            self._analysis_in_progress = True

        self._executor.submit(self._analyze_background, stats, iteration, best_score)

    def get_state(self) -> dict:
        """Return a thread-safe snapshot of current state."""
        with self._lock:
            return {
                "active_feedback": self._active_feedback,
                "artifact_summary": getattr(self, "_latest_artifact_summary", ""),
                "cost_spent": self._cost_spent,
                "budget_max": self._config.max_cost_dollars,
                "budget_exhausted": self._budget_exhausted,
                "analysis_in_progress": self._analysis_in_progress,
                "last_analysis_iter": self._last_analysis_iter,
            }

    def get_feedback_log(self) -> list:
        """Return a copy of the feedback audit log."""
        with self._lock:
            return list(self._feedback_log)

    def stop(self) -> None:
        """Shut down the background executor."""
        self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Background analysis (runs in ThreadPoolExecutor)
    # ------------------------------------------------------------------

    def _analyze_background(self, stats: dict, iteration: int, best_score: float) -> None:
        """Run LLM analysis and store result for main-thread pickup."""
        logger.info("AI feedback background analysis starting (iter=%d)", iteration)
        try:
            with self._lock:
                prev_feedback = self._previous_feedback
            context = self._build_context(stats, iteration)
            system = self._build_system_prompt(prev_feedback)

            response = self._call_llm(
                model=self._config.model,
                system=system,
                user_message=context,
                api_key=self._config.api_key or "",
                api_base=self._config.api_base,
                max_tokens=1024,
                timeout=self._config.api_timeout_seconds,
            )

            # Estimate cost (conservative: char/3 overestimates token count)
            input_chars = len(system) + len(context)
            output_chars = len(response)
            input_tokens = input_chars / _AVG_CHARS_PER_TOKEN
            output_tokens = output_chars / _AVG_CHARS_PER_TOKEN
            call_cost = (input_tokens / 1000) * _COST_PER_1K_INPUT + (
                output_tokens / 1000
            ) * _COST_PER_1K_OUTPUT

            with self._lock:
                self._cost_spent += call_cost
                cost_total = self._cost_spent

                if self._cost_spent >= self._config.max_cost_dollars:
                    self._budget_exhausted = True
                    logger.warning(
                        "AI feedback budget exhausted ($%.2f / $%.2f). Disabling.",
                        self._cost_spent,
                        self._config.max_cost_dollars,
                    )

            feedback_text = response.strip()

            # Quality gates
            if not feedback_text or len(feedback_text) < 10:
                logger.debug("AI feedback too short, skipping")
                return

            # "No changes needed" is a valid response -- clear active feedback
            if _is_no_changes(feedback_text):
                with self._lock:
                    self._pending_feedback = ""
                    self._feedback_log.append({
                        "iteration": iteration,
                        "timestamp": _time.time(),
                        "action": "no_changes",
                        "cost": call_cost,
                    })
                    if len(self._feedback_log) > _MAX_FEEDBACK_LOG:
                        self._feedback_log = self._feedback_log[-_MAX_FEEDBACK_LOG:]
                logger.info("AI analysis: no changes needed at iteration %d", iteration)
                return

            # Dedup: skip if nearly identical to previous
            if prev_feedback and self._is_similar(feedback_text, prev_feedback):
                logger.debug("AI feedback too similar to previous, skipping")
                return

            # Truncate and store
            feedback_text = feedback_text[: self._config.max_feedback_chars]
            with self._lock:
                self._pending_feedback = feedback_text
                self._previous_feedback = feedback_text

                # Reset auto-clear tracker
                self._iterations_with_feedback = 0
                self._best_score_at_feedback_start = best_score

                # Audit log
                self._feedback_log.append({
                    "iteration": iteration,
                    "timestamp": _time.time(),
                    "action": "generated",
                    "text": feedback_text[:200],
                    "cost": call_cost,
                })
                if len(self._feedback_log) > _MAX_FEEDBACK_LOG:
                    self._feedback_log = self._feedback_log[-_MAX_FEEDBACK_LOG:]

            logger.info(
                "AI feedback generated at iteration %d (%d chars, cost=$%.4f, total=$%.4f)",
                iteration,
                len(feedback_text),
                call_cost,
                cost_total,
            )

        except Exception as exc:
            logger.warning("AI feedback analysis failed: %s", exc, exc_info=True)
        finally:
            with self._lock:
                self._analysis_in_progress = False
                self._last_analysis_iter = iteration

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _should_analyze(self, iteration: int) -> bool:
        """Check all guardrails before triggering analysis. Must hold self._lock."""
        if self._analysis_in_progress:
            return False
        if self._budget_exhausted:
            return False
        if iteration - self._last_analysis_iter < self._config.interval:
            return False
        return True

    def _build_system_prompt(self, previous_feedback: str = "") -> str:
        """Build the system prompt for the analysis LLM call."""
        prompt = _ANALYSIS_SYSTEM_PROMPT
        if self._config.include_problem_description and self._problem_description:
            # Include first 1500 chars of problem description for domain context
            desc = self._problem_description[:1500]
            prompt += f"\n\nProblem context:\n{desc}"
        if previous_feedback:
            prompt += f"\n\nYour previous guidance was:\n{previous_feedback[:500]}"
        return prompt

    def _build_context(self, stats: dict, iteration: int) -> str:
        """Build a human-readable context string from database statistics."""
        parts = [f"=== Discovery State at Iteration {iteration} ==="]
        latest_artifact_summary = stats.get("latest_artifact_summary", "")
        if latest_artifact_summary and self._config.include_artifact_summary:
            latest_artifact_summary = str(latest_artifact_summary)[: self._config.max_artifact_summary_chars]
            parts.append(f"Latest evaluator artifact summary:\n{latest_artifact_summary}")
            with self._lock:
                self._latest_artifact_summary = latest_artifact_summary

        # Score distribution
        score_summary = stats.get("solution_score_summary", {})
        if score_summary:
            parts.append(
                f"Best score: {score_summary.get('best', '?')}\n"
                f"Q75: {score_summary.get('q75', '?')}, "
                f"Median: {score_summary.get('q50', '?')}, "
                f"Q25: {score_summary.get('q25', '?')}\n"
                f"Unique scores: {score_summary.get('unique_scores', '?')}"
            )

        # Population
        parts.append(f"Population size: {stats.get('population_size', '?')}")

        # Stagnation and diversity
        recent = stats.get("recent_solution_stats", {})
        if recent:
            stagnation = recent.get("iterations_without_improvement", 0)
            reuse_ratio = recent.get("most_reused_parent_ratio", 0)
            parts.append(
                f"Iterations without improvement: {stagnation}\n"
                f"Most-reused parent ratio: {reuse_ratio:.2f}"
            )

        # Score trajectory
        trajectory = recent.get("score_trajectory", [])
        if trajectory:
            recent_scores = trajectory[-15:]
            formatted = [f"{s:.6f}" if isinstance(s, float) else str(s) for s in recent_scores]
            parts.append(f"Recent score trajectory (last {len(recent_scores)}): {formatted}")

        # Top scores
        top_scores = stats.get("top_solution_scores", [])
        if top_scores:
            parts.append(f"Top scores: {top_scores[:5]}")

        return "\n\n".join(parts)

    @staticmethod
    def _is_similar(a: str, b: str, threshold: float = 0.8) -> bool:
        """Quick similarity check based on shared words (Jaccard on word tokens)."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return False
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union) > threshold
