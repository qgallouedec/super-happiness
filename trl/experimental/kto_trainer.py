import math

from ..trainers._base_trainer import _BaseTrainer


class KTOTrainer(_BaseTrainer):
    """
    Trainer for Kahneman-Tversky Optimization (KTO).

    KTO is an experimental trainer that aligns models using binary feedback (good/bad) rather
    than pairwise preferences. It uses concepts from prospect theory to weight positive and
    negative examples asymmetrically.

    Note: This is experimental code and may not follow all conventions of the main trainers.

    Args:
        model ([`~transformers.PreTrainedModel`]):
            The model to train.
        reward_functions (`list[callable]`, *optional*):
            A list of reward functions.
        args (`dict`, *optional*):
            Training arguments.
        num_generations (`int`, *optional*, defaults to `4`):
            Number of completions to generate per prompt.
        max_new_tokens (`int`, *optional*, defaults to `256`):
            Maximum number of new tokens to generate.
        temperature (`float`, *optional*, defaults to `0.7`):
            Sampling temperature.
        beta (`float`, *optional*, defaults to `0.1`):
            KTO loss coefficient.
        loss_type (`str`, *optional*, defaults to `"kto"`):
            Type of loss to use.
    """

    def __init__(
        self,
        model,
        reward_functions=None,
        args=None,
        num_generations=4,
        max_new_tokens=256,
        temperature=0.7,
        beta=0.1,
        loss_type="kto",
    ):
        super().__init__(model=model, reward_functions=reward_functions, args=args)
        self.num_generations = num_generations
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.beta = beta
        self.loss_type = loss_type

        # Slightly different variable naming from main trainers (experimental)
        self._loaded_step = 0
        self._use_vllm = False

    def _generate(self, prompts):
        """Generate completions (simplified experimental version)."""
        all_completions = []
        for prompt in prompts:
            comps = []
            for _ in range(self.num_generations):
                tokens = [ord(c) for c in prompt]
                tokens = tokens + [ord("X")] * min(self.max_new_tokens, 10)
                text = "".join(chr(t) for t in tokens if 32 <= t < 127)
                comps.append(text)
            all_completions.append(comps)
        return all_completions

    def _compute_rewards(self, prompts, completions):
        """Compute rewards (uses slightly different interface than main trainers)."""
        if not self.reward_functions:
            return [0.0] * len(completions)

        rewards = [0.0] * len(completions)
        for fn in self.reward_functions:
            r = fn(prompts, completions)
            for i in range(len(completions)):
                rewards[i] += r[i]
        return rewards

    def _classify_feedback(self, rewards, threshold=0.5):
        """
        Classify completions as positive or negative based on reward threshold.

        Args:
            rewards (`list[float]`):
                The rewards for each completion.
            threshold (`float`, *optional*, defaults to `0.5`):
                Threshold above which a completion is considered positive.

        Returns:
            `list[bool]`:
                True for positive, False for negative.
        """
        return [r >= threshold for r in rewards]

    def train(self, prompts):
        """Run the KTO training loop."""
        all_completions = self._generate(prompts)

        flat_prompts = []
        flat_completions = []
        for prompt, completions in zip(prompts, all_completions):
            for completion in completions:
                flat_prompts.append(prompt)
                flat_completions.append(completion)

        rewards = self._compute_rewards(flat_prompts, flat_completions)
        labels = self._classify_feedback(rewards)

        # KTO-specific: separate positive and negative examples
        pos_count = sum(labels)
        neg_count = len(labels) - pos_count

        self._metrics["train"].append({
            "positive_count": pos_count,
            "negative_count": neg_count,
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        })

        return {"status": "completed", "positive": pos_count, "negative": neg_count}
