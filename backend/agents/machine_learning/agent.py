"""DATORYX MachineLearning Agent - Builds and deploys machine learning models (LLM-backed)"""
from typing import Dict, Any, List

from ..base import BaseAgent, AgentCapability

SYSTEM_PROMPT = (
    "You are a machine learning engineer inside DATORYX. You recommend concrete, dataset-appropriate "
    "modeling choices. You are honest about the difference between a recommendation/plan and an actual "
    "trained model result - this system has no real training backend wired up, so you never claim a model "
    "has been fit or report fabricated accuracy numbers."
)


class MachineLearningAgent(BaseAgent):
    """Builds and deploys machine learning models"""

    def __init__(self, llm: Any = None):
        super().__init__(name="MachineLearning", description="Builds and deploys machine learning models", llm=llm)

    def _register_capabilities(self):
        self.capabilities = [
            AgentCapability("preprocess", "Preprocess datasets", {"data": "list", "steps": "list"}),
            AgentCapability("train_model", "Train ML models", {"X": "list", "y": "list"}),
            AgentCapability("cross_validate", "Cross-validate models", {"X": "list", "y": "list"}),
            AgentCapability("feature_engineering", "Engineer new features", {"data": "list", "operations": "list"}),
            AgentCapability("hyperparameter_tune", "Optimize hyperparameters", {"param_grid": "dict"})
        ]

    def _register_tools(self):
        self.tools = {
            "preprocess": self._preprocess,
            "train_model": self._train_model,
            "cross_validate": self._cross_validate,
            "feature_engineering": self._feature_engineering,
            "hyperparameter_tune": self._hyperparameter_tune
        }

    def _preprocess(self, data: List[Dict], steps: List[str]) -> Dict:
        # Preprocessing transforms are exact, well-defined operations - apply them directly
        # rather than asking the LLM to do arithmetic.
        processed = [dict(row) for row in data]
        for step in steps:
            if step == "normalize" and processed:
                numeric = {k: [r[k] for r in processed if isinstance(r.get(k), (int, float))]
                           for k in processed[0].keys()}
                for k, vals in numeric.items():
                    if vals:
                        min_v, max_v = min(vals), max(vals)
                        for r in processed:
                            if k in r and isinstance(r[k], (int, float)) and max_v != min_v:
                                r[k] = (r[k] - min_v) / (max_v - min_v)
            elif step == "encode_categorical" and processed:
                for k in processed[0].keys():
                    unique_vals = list(set(str(r.get(k, "")) for r in processed))
                    for r in processed:
                        if k in r:
                            r[f"{k}_encoded"] = unique_vals.index(str(r[k]))
            elif step == "handle_missing":
                for r in processed:
                    for k, v in r.items():
                        if v is None or v == "":
                            r[k] = 0 if isinstance(v, (int, float)) else "unknown"
        return {"records": len(processed), "steps_applied": steps, "sample": processed[:3]}

    def _train_model(self, X: List[List[float]], y: List,
                      algorithm: str = "random_forest") -> Dict:
        user_prompt = (
            f"Recommend hyperparameters for a '{algorithm}' model trained on {len(X)} samples with "
            f"{len(X[0]) if X else 0} features, predicting a target with {len(set(map(str, y)))} distinct "
            "values.\n\n"
            'Return JSON: {"hyperparameters": {...}, "rationale": "...", "risks": ["<e.g. overfitting, '
            'class imbalance, given these sample/feature counts>"]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "algorithm": algorithm,
            "hyperparameters": result.get("hyperparameters", {}),
            "rationale": result.get("rationale", ""),
            "risks": result.get("risks", []),
            "features": len(X[0]) if X else 0,
            "samples": len(X),
            "status": "recommendation only - no training backend connected; nothing was actually fit"
        }

    def _cross_validate(self, X: List[List[float]], y: List,
                         k_folds: int = 5) -> Dict:
        # Fold sizing is exact arithmetic; there is no real model to score, so we report
        # the split plan honestly instead of fabricating per-fold accuracy numbers.
        fold_size = len(X) // k_folds if k_folds else len(X)
        folds = []
        for i in range(k_folds):
            start = i * fold_size
            end = start + fold_size if i < k_folds - 1 else len(X)
            val_size = end - start
            folds.append({"fold": i + 1, "train_size": len(X) - val_size, "val_size": val_size})
        return {
            "folds": folds,
            "status": "split plan only - no training backend connected; no accuracy was actually measured"
        }

    def _feature_engineering(self, data: List[Dict], operations: List[Dict]) -> Dict:
        user_prompt = (
            f"Given a dataset with columns {list(data[0].keys()) if data else []}, propose concrete "
            f"engineered feature names and rationale for these requested operations: {operations}\n\n"
            'Return JSON: {"new_features": [{"name": "...", "rationale": "..."}]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        new_features = result.get("new_features", [])
        return {
            "new_features": new_features,
            "total_features": (len(data[0].keys()) if data else 0) + len(new_features)
        }

    def _hyperparameter_tune(self, param_grid: Dict[str, List],
                              scoring: str = "accuracy") -> Dict:
        import itertools
        combinations = list(itertools.product(*param_grid.values())) if param_grid else []

        user_prompt = (
            f"Given this hyperparameter grid: {param_grid}, optimizing for '{scoring}', which region of "
            f"the grid is most likely to perform well based on general ML practice? There are "
            f"{len(combinations)} total combinations, and no actual search has been run.\n\n"
            'Return JSON: {"suggested_starting_point": {...}, "search_strategy_advice": "..."}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "total_combinations": len(combinations),
            "parameter_grid": param_grid,
            "scoring_metric": scoring,
            "search_strategy": "grid_search",
            "suggested_starting_point": result.get("suggested_starting_point", {}),
            "search_strategy_advice": result.get("search_strategy_advice", ""),
            "status": "advisory only - no actual hyperparameter search was run"
        }
