from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class EvaluationResult:
    """
    Result of program evaluation containing both metrics and optional artifacts
    """

    metrics: Dict[str, float]
    artifacts: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        if not isinstance(data, dict):
            return cls(metrics={"error": 0.0}, artifacts={"normalization_error": str(type(data))})

        if isinstance(data.get("metrics"), dict):
            metrics = {
                k: float(v)
                for k, v in data.get("metrics", {}).items()
                if isinstance(v, (int, float))
            }
            artifacts = dict(data.get("artifacts", {})) if isinstance(data.get("artifacts"), dict) else {}
            for key, value in data.items():
                if key in ("metrics", "artifacts"):
                    continue
                if isinstance(value, (int, float)):
                    metrics.setdefault(key, float(value))
                else:
                    artifacts.setdefault(key, value)
            return cls(metrics=metrics, artifacts=artifacts)

        metrics: Dict[str, float] = {}
        artifacts: Dict[str, Any] = {}
        nested_artifacts = data.get("artifacts")
        if isinstance(nested_artifacts, dict):
            artifacts.update(nested_artifacts)

        for key, value in data.items():
            if key == "artifacts":
                continue
            if isinstance(value, (int, float)):
                metrics[key] = float(value)
            else:
                artifacts[key] = value

        return cls(metrics=metrics, artifacts=artifacts)

    def to_dict(self) -> Dict[str, Any]:
        result = dict(self.metrics)
        if self.artifacts:
            result["artifacts"] = self.artifacts
        return result
