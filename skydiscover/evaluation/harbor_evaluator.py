"""Harbor evaluator: runs Harbor-format tasks inside a persistent Docker container.

Harbor tasks use a different container protocol from the standard
ContainerizedEvaluator:

  - Solution is injected at a task-specific path (e.g. ``/app/solver.py``)
    extracted from ``solution/solve.sh`` or ``instruction.md``.
  - Evaluation runs ``tests/test.sh`` instead of ``evaluate.sh``.
  - The reward is read from ``/logs/verifier/reward.txt`` (float) or
    ``/logs/verifier/reward.json`` (dict) instead of JSON on stdout.

A Harbor task directory has this structure::

    task_dir/
    ├── task.toml              # metadata, timeouts
    ├── instruction.md         # problem description (shown to LLM)
    ├── environment/
    │   └── Dockerfile
    ├── tests/
    │   ├── test.sh            # verification entrypoint
    │   └── ...                # supporting test files
    └── solution/              # reference solution (optional, never shown to LLM)
        └── solve.sh

See https://harborframework.com/docs for the full specification.
"""

import json
import logging
import os
import re
import subprocess

from skydiscover.evaluation.container_evaluator import ContainerizedEvaluator
from skydiscover.evaluation.evaluation_result import EvaluationResult

logger = logging.getLogger(__name__)

# Most common solution path across Harbor benchmarks — used as fallback.
_DEFAULT_SOLUTION_PATH = "/app/solution.py"
_MAX_REWARD_PAYLOAD_CHARS = 4000


class HarborEvaluator(ContainerizedEvaluator):
    """Evaluates programs using the Harbor container protocol.

    Extends ContainerizedEvaluator, overriding only the container interaction
    methods: image building, solution injection, test execution, and reward
    reading.
    """

    def __init__(self, benchmark_dir, config, max_concurrent=4):
        self.task_dir = os.path.abspath(benchmark_dir)
        self.solution_path = self._extract_solution_path()
        scaffold_solution_path = self._extract_scaffold_runner_solution_path()
        if scaffold_solution_path:
            self.solution_path = scaffold_solution_path
        self._tests_uploaded = False
        self._apply_task_toml_timeout(config)
        super().__init__(benchmark_dir, config, max_concurrent)
        self._init_container()

    # ------------------------------------------------------------------
    # Override: image building
    # ------------------------------------------------------------------

    def _build_image(self) -> str:
        """Build from environment/Dockerfile."""
        name = os.path.basename(os.path.normpath(self.task_dir))
        tag = f"skydiscover-harbor-{name}:latest"
        dockerfile_dir = os.path.join(self.task_dir, "environment")

        logger.info(f"Building Harbor image: {tag} (from {dockerfile_dir})")
        result = subprocess.run(
            ["docker", "build", "-t", tag, dockerfile_dir],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Docker build failed for {dockerfile_dir}:\n{result.stderr}")
        return tag

    # ------------------------------------------------------------------
    # Override: container interaction
    # ------------------------------------------------------------------

    def _run_container(self, program_solution: str, mode: str) -> EvaluationResult:
        """Inject solution, run tests, read reward."""
        # Clear stale reward files from previous evaluations.
        self._exec("rm -f /logs/verifier/reward.txt /logs/verifier/reward.json")

        # Reset workspace and /app to fixture state so each evaluation starts clean.
        self._exec("rm -rf /workspace && mkdir -p /workspace")
        self._exec("tar xzf /tmp/app_fixture.tar.gz -C / 2>/dev/null || true")

        # Ensure parent directory exists and inject solution.
        parent_dir = os.path.dirname(self.solution_path)
        if parent_dir:
            self._exec(f"mkdir -p '{parent_dir}'")
        inject = subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                self.container_id,
                "/bin/sh",
                "-c",
                f"cat > '{self.solution_path}'",
            ],
            input=program_solution.encode(),
            capture_output=True,
        )
        if inject.returncode != 0:
            logger.error(f"Failed to inject solution: {inject.stderr.decode()}")
            return EvaluationResult(
                metrics={"combined_score": 0.0},
                artifacts={"error": f"injection failed: {inject.stderr.decode()}"},
            )

        try:
            # Run tests.
            proc = subprocess.run(
                [
                    "docker",
                    "exec",
                    self.container_id,
                    "bash",
                    "-c",
                    "chmod +x /tests/test.sh && /tests/test.sh",
                ],
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
            )

            # Read reward regardless of exit code — test.sh may exit non-zero
            # but still write a reward (e.g. partial credit).
            result = self._read_reward(proc.stdout, proc.stderr)

            if proc.returncode != 0:
                result.artifacts.setdefault("test_exit_code", str(proc.returncode))
            if proc.stderr.strip():
                result.artifacts.setdefault("stderr", proc.stderr)
            if proc.stdout.strip():
                result.artifacts.setdefault("stdout", proc.stdout)

            return result
        except subprocess.TimeoutExpired:
            logger.error(f"docker exec timed out after {self.config.timeout}s")
            return EvaluationResult(
                metrics={"combined_score": 0.0},
                artifacts={"error": f"docker exec timed out after {self.config.timeout}s"},
            )

        finally:
            # Clean up solution so the container is fresh for next evaluation.
            self._exec(f"rm -f '{self.solution_path}'")

    # ------------------------------------------------------------------
    # Harbor-specific helpers
    # ------------------------------------------------------------------

    def _apply_task_toml_timeout(self, config) -> None:
        """Read verifier.timeout_sec from task.toml and apply it to config."""
        toml_path = os.path.join(self.task_dir, "task.toml")
        if not os.path.exists(toml_path):
            return
        try:
            with open(toml_path) as f:
                text = f.read()
            match = re.search(r"timeout_sec\s*=\s*(\d+)", text)
            if match:
                timeout_s = int(match.group(1))
                max_timeout_raw = os.getenv("TBENCH_MAX_TASK_TIMEOUT", "").strip()
                if max_timeout_raw.isdigit() and int(max_timeout_raw) > 0:
                    timeout_s = min(timeout_s, int(max_timeout_raw))
                config.timeout = timeout_s
                logger.info(f"Harbor task.toml: set evaluator timeout to {config.timeout}s")
        except Exception as e:
            logger.warning(f"Failed to read task.toml: {e}")

    def _init_container(self):
        """Create log directories and upload test files into the container."""
        self._exec("mkdir -p /logs/verifier /logs/agent /logs/artifacts")

        # Snapshot /app after Dockerfile setup so we can reset it before each eval.
        self._exec("tar czf /tmp/app_fixture.tar.gz -C / app 2>/dev/null || true")

        # Upload the tests/ directory.
        tests_dir = os.path.join(self.task_dir, "tests")
        if os.path.isdir(tests_dir):
            self._exec("rm -rf /tests")
            subprocess.run(
                ["docker", "cp", tests_dir, f"{self.container_id}:/tests"],
                capture_output=True,
                check=True,
            )
            self._tests_uploaded = True
            logger.debug("Uploaded tests/ to container")
        else:
            raise RuntimeError(f"No tests/ directory found in {self.task_dir}")

        # Upload instruction.md so test.sh can read it at /task/instruction.md
        instruction_file = os.path.join(self.task_dir, "instruction.md")
        if os.path.exists(instruction_file):
            self._exec("mkdir -p /task")
            subprocess.run(
                ["docker", "cp", instruction_file, f"{self.container_id}:/task/instruction.md"],
                capture_output=True,
                check=True,
            )
            logger.debug("Uploaded instruction.md to /task/instruction.md")

    def _read_reward(self, test_stdout: str = "", test_stderr: str = "") -> EvaluationResult:
        """Read the reward from /logs/verifier/reward.txt or reward.json."""
        for path, is_json in [
            ("/logs/verifier/reward.json", True),
            ("/logs/verifier/reward.txt", False),
        ]:
            proc = subprocess.run(
                ["docker", "exec", self.container_id, "cat", path],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0 or not proc.stdout.strip():
                continue

            try:
                if is_json:
                    data = json.loads(proc.stdout.strip())
                    raw = data.get("reward", data.get("score"))
                    if raw is None:
                        logger.warning(
                            "No 'reward' or 'score' key in %s; defaulting to 0",
                            path,
                        )
                        raw = 0
                    reward = float(raw)
                    metrics = {"combined_score": reward}
                    artifacts = {}
                    for k, v in data.items():
                        if isinstance(v, (int, float)) and k not in (
                            "reward",
                            "score",
                        ):
                            metrics[k] = float(v)
                        elif k not in ("reward", "score"):
                            artifacts[k] = v
                    if artifacts:
                        artifacts.setdefault(
                            "reward_payload",
                            proc.stdout.strip()[:_MAX_REWARD_PAYLOAD_CHARS],
                        )
                    return EvaluationResult(metrics=metrics, artifacts=artifacts)
                else:
                    reward = float(proc.stdout.strip())
                    return EvaluationResult(
                        metrics={"combined_score": reward},
                        artifacts={
                            "reward_payload": proc.stdout.strip()[:_MAX_REWARD_PAYLOAD_CHARS]
                        },
                    )
            except (ValueError, json.JSONDecodeError, StopIteration) as e:
                logger.warning(f"Failed to parse reward from {path}: {e}")
                continue

        logger.error("No reward file found in /logs/verifier/")
        return EvaluationResult(
            metrics={"combined_score": 0.0},
            artifacts={
                "error": "no reward file written by test.sh",
                "test_stdout": test_stdout,
                "test_stderr": test_stderr,
            },
        )

    def _extract_solution_path(self) -> str:
        """Extract the expected solution file path for this Harbor task.

        Uses a three-tier strategy (most reliable first):

        1. **Parse ``solution/solve.sh``** — the authoritative reference solution
           script almost always contains a ``cat > /path/to/file`` redirect that
           reveals the exact injection path.
        2. **Parse ``instruction.md``** — look for explicit absolute paths in
           backticks or after prepositions like "in", "at", "to".
        3. **Default to ``/app/solution.py``** — the most common path across
           Harbor benchmarks (evoeval, livecodebench, usaco, etc.).
        """
        # Tier 1: parse solution/solve.sh (most reliable).
        path = self._extract_path_from_solve_sh()
        if path:
            logger.info(f"Extracted solution path from solve.sh: {path}")
            return path

        # Tier 2: parse instruction.md.
        path = self._extract_path_from_instruction()
        if path:
            logger.info(f"Extracted solution path from instruction.md: {path}")
            return path

        # Tier 3: default.
        logger.warning(f"Could not extract solution path, using default: {_DEFAULT_SOLUTION_PATH}")
        return _DEFAULT_SOLUTION_PATH

    def _extract_scaffold_runner_solution_path(self) -> str:
        """Detect explicit scaffold runner import path from tests/test.sh."""
        test_sh = os.path.join(self.task_dir, "tests", "test.sh")
        if not os.path.exists(test_sh):
            return ""
        try:
            with open(test_sh) as f:
                text = f.read()
        except Exception:
            return ""
        match = re.search(
            r'spec_from_file_location\(\s*["\']solution["\']\s*,\s*["\'](/[^"\']+)["\']\s*\)',
            text,
        )
        if not match:
            return ""
        path = self._sanitize_candidate_path(match.group(1))
        if path:
            logger.info(f"Using scaffold runner solution path from tests/test.sh: {path}")
        return path

    def _sanitize_candidate_path(self, path: str) -> str:
        cleaned = (path or "").strip().strip("`'\"")
        # Reject shell fragments and unresolved variables from solve.sh parsing.
        if not cleaned.startswith("/"):
            return ""
        if "$" in cleaned or any(ch in cleaned for ch in (";", "|", "\n", "\r")):
            return ""
        return cleaned

    def _extract_path_from_solve_sh(self) -> str:
        """Extract the solution target path from ``solution/solve.sh``.

        Looks for shell redirect patterns like ``cat > /app/solver.py``
        or ``> /workspace/solution.py``.  If the path is relative, resolves
        it against the last ``cd`` target found before the redirect.
        """
        solve_sh = os.path.join(self.task_dir, "solution", "solve.sh")
        if not os.path.exists(solve_sh):
            return ""

        try:
            with open(solve_sh) as f:
                text = f.read()
        except Exception:
            return ""

        _CODE_EXTS = r"\.(?:py|sh|js|ts|cpp|c|rs|go|java|rb)"

        # First try: absolute path redirects.
        for pattern in [
            rf"cat\s+>\s*(/\S+{_CODE_EXTS})",
            rf">\s*(/\S+{_CODE_EXTS})",
        ]:
            match = re.search(pattern, text)
            if match:
                return self._sanitize_candidate_path(match.group(1))

        # Second try: relative path redirect (e.g. crustbench writes to
        # src/interfaces/base122.rs after cd-ing into a project directory).
        redirect_pattern = rf"cat\s+>\s*(\S+{_CODE_EXTS})"
        redirect_match = re.search(redirect_pattern, text)
        if redirect_match:
            rel_path = redirect_match.group(1)

            # Resolve the base directory.  Strategy:
            # 1. Look for concrete absolute paths in cd commands.
            # 2. Look for absolute paths assigned to shell variables (the
            #    variable may be used with cd later — e.g. RBENCH_DIR).
            # 3. Fall back to the Dockerfile WORKDIR.
            candidates = re.findall(r'cd\s+"?(/[^"$\s]+)"?\s*$', text, re.MULTILINE)
            if not candidates:
                # Variable assignments like RBENCH_DIR="/workspace/rbench_reference"
                candidates = re.findall(r'[A-Z_]+=\s*"?(/[^"$\s]+)"?\s*$', text, re.MULTILINE)

            if candidates:
                base = candidates[0].rstrip('"')
            else:
                # Dockerfile WORKDIR fallback.
                base = "/workspace"
                dockerfile = os.path.join(self.task_dir, "environment", "Dockerfile")
                if os.path.exists(dockerfile):
                    try:
                        with open(dockerfile) as f:
                            for line in f:
                                m = re.match(r"WORKDIR\s+(/\S+)", line)
                                if m:
                                    base = m.group(1)
                    except Exception:
                        pass

            return self._sanitize_candidate_path(os.path.join(base, rel_path))

        return ""

    def _extract_path_from_instruction(self) -> str:
        """Extract the solution file path from ``instruction.md``."""
        instruction_path = os.path.join(self.task_dir, "instruction.md")
        if not os.path.exists(instruction_path):
            return ""

        try:
            with open(instruction_path) as f:
                text = f.read()
        except Exception:
            return ""

        patterns = [
            r'[`"\'](/\S+\.(?:py|sh|js|ts|cpp|c|rs|go|java))[`"\']',
            r"(?:in|at|to|into)\s+(/\S+\.(?:py|sh|js|ts|cpp|c|rs|go|java))",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return self._sanitize_candidate_path(match.group(1))

        return ""

    def _exec(self, cmd: str) -> subprocess.CompletedProcess:
        """Run a shell command inside the container."""
        return subprocess.run(
            ["docker", "exec", self.container_id, "/bin/sh", "-c", cmd],
            capture_output=True,
            text=True,
        )
