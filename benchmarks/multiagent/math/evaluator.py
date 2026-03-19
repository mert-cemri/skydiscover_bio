"""
Accuracy-based evaluator for multi-agent math workflows.

Runs the candidate program on MATH-500 problems and scores by fraction correct.
No taxonomy, no LLM judge — pure accuracy signal.

Environment variables:
  SOLVER_MODEL       — model for agents (default: gpt-4o-mini)
  NUM_PROBLEMS       — problems per evaluation (default: 10)
  CANDIDATE_TIMEOUT_S — per-question timeout in seconds (default: 120)
"""

import os, sys, re, time, json, pickle, tempfile, subprocess, traceback, shutil
import random
from typing import List, Dict, Any


# ═══════════════════════════════════════════════════════════════════════════
# Dataset helpers
# ═══════════════════════════════════════════════════════════════════════════

_CACHED_DATASET = None


def _load_bench(num_problems: int = 20) -> List[Dict[str, Any]]:
    """Load MATH-500 problems from local cache or download."""
    global _CACHED_DATASET
    cache_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "olympiad_cache.json"
    )

    if _CACHED_DATASET is not None and len(_CACHED_DATASET) >= num_problems:
        return _CACHED_DATASET[:num_problems]

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            _CACHED_DATASET = json.load(f)
            return _CACHED_DATASET[:num_problems]

    from datasets import load_dataset

    print("Downloading MATH-500 dataset...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    problems = []
    for item in dataset:
        question = item.get("problem", "")
        solution = item.get("solution", "")
        boxed = re.search(r"\\boxed\{([^}]*)\}", solution)
        answer = boxed.group(1) if boxed else solution.strip()
        if question and answer:
            problems.append({"question": question, "answer": str(answer)})

    random.seed(42)
    sampled = random.sample(problems, min(num_problems, len(problems)))
    _CACHED_DATASET = sampled

    with open(cache_file, "w") as f:
        json.dump(sampled, f, indent=2)
    print(f"Cached {len(sampled)} problems to {cache_file}")
    return sampled


def _normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison."""
    if not answer:
        return ""
    answer = re.sub(r"FINAL ANSWER:\s*", "", answer, flags=re.IGNORECASE)
    boxed = re.search(r"\\boxed\{([^}]*)\}", answer)
    if boxed:
        answer = boxed.group(1)
    answer = answer.replace("$", "")
    answer = " ".join(answer.split())
    while answer and answer[-1] in [".", ","]:
        answer = answer[:-1].strip()
    return answer.lower()


def _is_correct(pred, gold) -> bool:
    """Check if a predicted answer matches the gold answer."""
    p = _normalize_answer(str(pred) if pred else "")
    g = _normalize_answer(str(gold) if gold else "")
    if not p or not g:
        return False
    if p == g:
        return True
    if p in g or g in p:
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Subprocess runner — isolates candidate execution
# ═══════════════════════════════════════════════════════════════════════════


class CandidateTimeout(Exception):
    pass


def _run_candidate(
    program_path: str, question: str, timeout_seconds: int = 120
) -> Dict:
    """Run the candidate program in a subprocess for isolation."""
    program_path_safe = program_path.replace("\\", "/")
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as tmp:
        res_path = tmp.name + ".res"
        res_path_safe = res_path.replace("\\", "/")
        script = f"""
import os, sys, pickle, traceback, json
sys.path.insert(0, os.path.dirname('{program_path_safe}'))
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("prog", '{program_path_safe}')
    prog = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prog)
    out = prog.run_agentic({json.dumps(question)})
    with open('{res_path_safe}', 'wb') as f:
        pickle.dump({{'out': out}}, f)
except Exception as e:
    traceback.print_exc()
    with open('{res_path_safe}', 'wb') as f:
        pickle.dump({{'error': str(e)}}, f)
"""
        tmp.write(script)
        tmp_path = tmp.name

    try:
        env = os.environ.copy()
        p = subprocess.Popen(
            [sys.executable, tmp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = p.communicate(timeout=timeout_seconds)
            if stderr:
                err_text = stderr.decode(errors="ignore")[:500]
                if err_text.strip():
                    print(f"  stderr: {err_text}")
            if p.returncode != 0:
                raise RuntimeError(f"exit code {p.returncode}")
            if not os.path.exists(res_path):
                raise RuntimeError("results file missing")
            with open(res_path, "rb") as f:
                results = pickle.load(f)
            if "error" in results:
                raise RuntimeError(results["error"])
            return results["out"]
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
            raise CandidateTimeout(f"timed out after {timeout_seconds}s")
    finally:
        for fp in (tmp_path, res_path):
            if os.path.exists(fp):
                os.unlink(fp)


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: Quick validation — does the program even run?
# ═══════════════════════════════════════════════════════════════════════════


def evaluate_stage1(program_path: str) -> Dict[str, Any]:
    """Quick check: run on 2 problems to verify the program executes."""
    start = time.time()

    stable_copy = program_path + ".eval_copy.py"
    try:
        shutil.copy2(program_path, stable_copy)
    except FileNotFoundError:
        return {
            "combined_score": 0.0,
            "error": f"Program file not found: {program_path}",
            "eval_wall_s": float(time.time() - start),
        }

    bench = _load_bench(num_problems=2)
    timeout = int(os.getenv("CANDIDATE_TIMEOUT_S", "120"))
    n_ok = 0

    try:
        for ex in bench:
            try:
                episode = _run_candidate(stable_copy, ex["question"], timeout)
                pred = episode.get("pred", "__NO_ANSWER__")
                if pred != "__NO_ANSWER__":
                    n_ok += 1
            except (CandidateTimeout, Exception) as e:
                print(f"  Stage 1 error: {e}")
                return {
                    "combined_score": 0.0,
                    "error": str(e),
                    "eval_wall_s": float(time.time() - start),
                }

        # If it runs and produces answers, give a minimum passing score
        return {
            "combined_score": 0.1 if n_ok > 0 else 0.0,
            "stage1_ok": n_ok,
            "eval_wall_s": float(time.time() - start),
        }
    finally:
        if os.path.exists(stable_copy):
            os.unlink(stable_copy)


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2: Full accuracy evaluation
# ═══════════════════════════════════════════════════════════════════════════


def evaluate_stage2(program_path: str) -> Dict[str, Any]:
    """Full evaluation: run on all problems, score by accuracy."""
    return _evaluate_full(program_path)


def evaluate(program_path: str) -> Dict[str, Any]:
    """Single-stage evaluator (used when cascade_evaluation is disabled)."""
    return _evaluate_full(program_path)


def _evaluate_full(program_path: str) -> Dict[str, Any]:
    """Run the candidate on NUM_PROBLEMS and return accuracy metrics."""
    start = time.time()

    stable_copy = program_path + ".eval_copy.py"
    try:
        shutil.copy2(program_path, stable_copy)
    except FileNotFoundError:
        return {
            "combined_score": 0.0,
            "error": f"Program file not found: {program_path}",
            "eval_wall_s": float(time.time() - start),
        }

    num_problems = int(os.getenv("NUM_PROBLEMS", "10"))
    bench = _load_bench(num_problems=num_problems)
    timeout = int(os.getenv("CANDIDATE_TIMEOUT_S", "120"))

    n_correct = 0
    n_verified = 0
    total_latency = 0.0
    num_agents = 0
    errors = []
    per_problem: List[Dict] = []

    try:
        for idx, ex in enumerate(bench):
            try:
                episode = _run_candidate(stable_copy, ex["question"], timeout)
            except CandidateTimeout as e:
                errors.append(f"Q{idx+1}: {e}")
                per_problem.append({"question_idx": idx, "correct": False, "error": "timeout"})
                continue
            except Exception as e:
                errors.append(f"Q{idx+1}: {e}")
                per_problem.append({"question_idx": idx, "correct": False, "error": str(e)})
                continue

            pred = episode.get("pred", "__NO_ANSWER__")
            gold = ex.get("answer", "")
            correct = _is_correct(pred, gold)

            if correct:
                n_correct += 1
            if episode.get("verified", False):
                n_verified += 1
            total_latency += float(episode.get("latency_s", 0.0))
            num_agents = max(num_agents, int(episode.get("num_agents", 0)))

            per_problem.append({
                "question_idx": idx,
                "correct": correct,
                "pred": _normalize_answer(str(pred)),
                "gold": _normalize_answer(gold),
            })

        n_attempted = len(bench)
        n_answered = sum(1 for p in per_problem if "error" not in p)
        accuracy = float(n_correct) / max(1, n_answered) if n_answered > 0 else 0.0

        # Penalize excessive agents (>10 agents gets -0.01 per extra)
        agent_penalty = max(0, (num_agents - 10)) * 0.01

        combined_score = max(0.0, accuracy - agent_penalty)

        return {
            "combined_score": float(combined_score),
            "accuracy": float(accuracy),
            "n_correct": int(n_correct),
            "n_attempted": int(n_attempted),
            "n_verified": int(n_verified),
            "num_agents": int(num_agents),
            "total_latency_s": float(total_latency),
            "avg_latency_s": float(total_latency / max(1, n_attempted)),
            "eval_wall_s": float(time.time() - start),
            "errors": errors[:5],  # cap error list
        }

    except Exception as e:
        traceback.print_exc()
        return {
            "combined_score": 0.0,
            "error": str(e),
            "eval_wall_s": float(time.time() - start),
        }
    finally:
        if os.path.exists(stable_copy):
            os.unlink(stable_copy)
