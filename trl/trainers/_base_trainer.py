import os
from collections import defaultdict


class _BaseTrainer:
    """
    Base class for all trainers. Provides only minimal utilities (model card generation).

    All training logic — generation, reward computation, metric logging, weight syncing — is
    deliberately duplicated in each trainer subclass for isolation and readability.

    Args:
        model (`torch.nn.Module`):
            The model to train.
        reward_functions (`list[callable]`, *optional*):
            A list of reward functions to use for computing rewards.
        args (`dict`, *optional*):
            Training arguments.
    """

    def __init__(self, model, reward_functions=None, args=None):
        self.model = model
        self.reward_functions = reward_functions or []
        self.args = args or {}
        self._metrics = defaultdict(list)

    def generate_model_card(self, output_dir: str | None = None):
        """
        Generate a model card for the trained model.

        Args:
            output_dir (`str`, *optional*):
                Directory where the model card will be saved.

        Returns:
            `str`:
                The model card content as a string.
        """
        card = f"# Model Card\n\n"
        card += f"## Trainer: {self.__class__.__name__}\n\n"
        card += f"## Training Args\n\n"
        for key, value in self.args.items():
            card += f"- **{key}**: {value}\n"

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "MODEL_CARD.md"), "w") as f:
                f.write(card)

        return card

    def _log_metrics(self, mode: str, metrics: dict):
        """
        Log metrics for a given mode.

        Args:
            mode (`str`):
                The mode to log metrics for (e.g., "train", "eval").
            metrics (`dict`):
                The metrics to log.
        """
        for key, value in metrics.items():
            self._metrics[mode].append({key: value})
