"""DATORYX DeepLearning Agent - Builds neural networks and deep learning models (LLM-backed)"""
from typing import Dict, Any, List

from ..base import BaseAgent, AgentCapability

SYSTEM_PROMPT = (
    "You are a deep learning architect inside DATORYX. You design PyTorch/TensorFlow-ready architectures "
    "and training configurations grounded in current best practice, tailored to the specific task given - "
    "not generic boilerplate."
)


class DeepLearningAgent(BaseAgent):
    """Builds neural networks and deep learning models"""

    def __init__(self, llm: Any = None):
        super().__init__(name="DeepLearning", description="Builds neural networks and deep learning models", llm=llm)

    def _register_capabilities(self):
        self.capabilities = [
            AgentCapability("design_architecture", "Design neural network architectures", {"task": "str", "input_shape": "list"}),
            AgentCapability("configure_training", "Configure training pipelines", {"epochs": "int"}),
            AgentCapability("evaluate_model", "Evaluate deep learning models", {"predictions": "list"}),
            AgentCapability("transfer_learning", "Apply transfer learning", {"base_model": "str"}),
            AgentCapability("deploy_model", "Deploy DL models", {"model_path": "str"})
        ]

    def _register_tools(self):
        self.tools = {
            "design_architecture": self._design_architecture,
            "configure_training": self._configure_training,
            "evaluate_model": self._evaluate_dl_model,
            "transfer_learning": self._transfer_learning,
            "deploy_model": self._deploy_model
        }

    def _design_architecture(self, task: str, input_shape: List[int],
                              complexity: str = "medium") -> Dict:
        user_prompt = (
            f"Design a neural network architecture for this task: {task}\n"
            f"Input shape: {input_shape}. Desired complexity: {complexity}.\n\n"
            'Return JSON: {"layers": [{"type": "...", ...layer params}], "optimizer": "...", "loss": "...", '
            '"total_params_estimate": "<e.g. 1.2M>", "framework": "PyTorch or TensorFlow", '
            '"rationale": "<why this architecture fits the task>"}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "task": task,
            "input_shape": input_shape,
            "architecture": {
                "layers": result.get("layers", []),
                "optimizer": result.get("optimizer", "adam"),
                "loss": result.get("loss", "")
            },
            "total_params_estimate": result.get("total_params_estimate", "unknown"),
            "framework": result.get("framework", "PyTorch/TensorFlow"),
            "rationale": result.get("rationale", "")
        }

    def _configure_training(self, epochs: int = 100, batch_size: int = 32,
                             learning_rate: float = 0.001) -> Dict:
        user_prompt = (
            f"Recommend a training configuration for {epochs} epochs, batch size {batch_size}, "
            f"initial learning rate {learning_rate}.\n\n"
            'Return JSON: {"scheduler": "...", "early_stopping": {"patience": <int>, "monitor": "..."}, '
            '"callbacks": ["..."], "augmentation": ["..."], "notes": "<why these choices>"}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "scheduler": result.get("scheduler", "ReduceLROnPlateau"),
            "early_stopping": result.get("early_stopping", {"patience": 10, "monitor": "val_loss"}),
            "callbacks": result.get("callbacks", []),
            "augmentation": result.get("augmentation", []),
            "notes": result.get("notes", "")
        }

    def _evaluate_dl_model(self, predictions: List, actuals: List,
                            task_type: str = "classification") -> Dict:
        # Metric computation is deterministic math, not a "stub" - keep it exact.
        if task_type == "classification":
            correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
            accuracy = correct / len(predictions) if predictions else 0
            metrics = {"accuracy": accuracy, "confusion_matrix": "computed_from_predictions"}
        else:
            n = len(predictions) or 1
            mse = sum((p - a) ** 2 for p, a in zip(predictions, actuals)) / n
            metrics = {"mse": mse, "rmse": mse ** 0.5}

        user_prompt = (
            f"A {task_type} model was evaluated with these metrics: {metrics}, over {len(predictions)} samples. "
            'Return JSON: {"interpretation": "<1-2 sentence assessment of whether this is good performance>", '
            '"recommendations": ["..."]}'
        )
        commentary = self._llm_json(SYSTEM_PROMPT, user_prompt)
        metrics["interpretation"] = commentary.get("interpretation", "")
        metrics["recommendations"] = commentary.get("recommendations", [])
        return metrics

    def _transfer_learning(self, base_model: str, target_classes: int) -> Dict:
        user_prompt = (
            f"Design a transfer learning plan using base model '{base_model}' adapted to {target_classes} "
            "output classes.\n\n"
            'Return JSON: {"pretrained_on": "...", "base_output_dim": <int>, "frozen_layers": "...", '
            '"new_head": {"input_dim": <int>, "output_dim": <int>}, "fine_tuning_strategy": "...", '
            '"rationale": "..."}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "base_model": base_model,
            "pretrained_on": result.get("pretrained_on", "unknown"),
            "frozen_layers": result.get("frozen_layers", "all except last 2"),
            "new_head": result.get("new_head", {"input_dim": 512, "output_dim": target_classes}),
            "fine_tuning_strategy": result.get("fine_tuning_strategy", "gradual_unfreezing"),
            "rationale": result.get("rationale", "")
        }

    def _deploy_model(self, model_path: str, platform: str = "docker") -> Dict:
        user_prompt = (
            f"Plan a deployment of a trained model at '{model_path}' onto platform '{platform}'.\n\n"
            'Return JSON: {"serving": "...", "batching": {"enabled": true/false, "max_batch_size": <int>}, '
            '"scaling": {"min_replicas": <int>, "max_replicas": <int>}, "monitoring": ["..."], "notes": "..."}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "model_path": model_path,
            "platform": platform,
            "serving": result.get("serving", "TorchServe/TF Serving"),
            "batching": result.get("batching", {"enabled": True, "max_batch_size": 32}),
            "scaling": result.get("scaling", {"min_replicas": 2, "max_replicas": 10}),
            "monitoring": result.get("monitoring", ["latency", "throughput", "error_rate"]),
            "notes": result.get("notes", "")
        }
