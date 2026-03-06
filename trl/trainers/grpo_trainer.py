import math
import torch

from ._base_trainer import _BaseTrainer


class GRPOTrainer(_BaseTrainer):
    """
    Trainer for Group Relative Policy Optimization (GRPO).

    GRPO eliminates the need for a critic model by estimating the baseline from group scores.
    For each prompt, multiple completions are generated and their rewards are normalized within
    the group to compute advantages.

    Args:
        model ([`~transformers.PreTrainedModel`]):
            The model to train.
        reward_functions (`list[callable]`, *optional*):
            A list of reward functions to use for computing rewards.
        args (`dict`, *optional*):
            Training arguments.

        > Parameters for generation:

        num_generations (`int`, *optional*, defaults to `4`):
            Number of completions to generate per prompt.
        max_new_tokens (`int`, *optional*, defaults to `256`):
            Maximum number of new tokens to generate.
        temperature (`float`, *optional*, defaults to `0.7`):
            Sampling temperature.

        > Parameters for training:

        num_train_epochs (`int`, *optional*, defaults to `1`):
            Number of training epochs.
        learning_rate (`float`, *optional*, defaults to `1e-6`):
            Learning rate.
        kl_coeff (`float`, *optional*, defaults to `0.1`):
            KL penalty coefficient.

    Examples:

    ```python
    >>> trainer = GRPOTrainer(model=model, reward_functions=[reward_fn])
    >>> trainer.train(prompts=["Solve 2+2"])
    ```
    """

    def __init__(
        self,
        model,
        reward_functions=None,
        args=None,
        num_generations: int = 4,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        num_train_epochs: int = 1,
        learning_rate: float = 1e-6,
        kl_coeff: float = 0.1,
    ):
        super().__init__(model=model, reward_functions=reward_functions, args=args)
        self.num_generations = num_generations
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.kl_coeff = kl_coeff

        # Track the last step at which we loaded weights from the vLLM engine
        self._last_loaded_step = 0
        # Track whether the vLLM engine is available
        self._vllm_available = False

    # ---------------------------------------------------------------------------
    # Generation
    # ---------------------------------------------------------------------------

    def _generate_single_turn(self, prompts: list[str]) -> list[list[str]]:
        """
        Generate completions for a list of prompts.

        For each prompt, we generate `self.num_generations` completions. If vLLM is available,
        we use the vLLM engine for fast generation. Otherwise, we fall back to standard
        autoregressive generation.

        Args:
            prompts (`list[str]`):
                A list of prompt strings.

        Returns:
            `list[list[str]]`:
                A list of lists of completion strings.
        """
        all_completions = []

        for prompt in prompts:
            completions = []

            if self._vllm_available:
                # Use vLLM engine for fast generation
                outputs = self._vllm_generate(
                    prompt,
                    n=self.num_generations,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )
                for output in outputs:
                    # GRPO extracts logprobs from vLLM outputs
                    completion_text = output["text"]
                    completion_logprobs = output["logprobs"]
                    completions.append(completion_text)
            else:
                # Fall back to standard autoregressive generation
                for _ in range(self.num_generations):
                    input_ids = self._tokenize(prompt)
                    output_ids = self._autoregressive_generate(
                        input_ids,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                    )
                    completion_text = self._decode(output_ids)
                    completions.append(completion_text)

            all_completions.append(completions)

        return all_completions

    def _vllm_generate(self, prompt, n, max_tokens, temperature):
        """
        Generate completions using the vLLM engine.

        Args:
            prompt (`str`):
                The prompt string.
            n (`int`):
                Number of completions to generate.
            max_tokens (`int`):
                Maximum number of tokens to generate.
            temperature (`float`):
                Sampling temperature.

        Returns:
            `list[dict]`:
                A list of dicts with keys "text" and "logprobs".
        """
        # Placeholder for vLLM integration
        raise NotImplementedError("vLLM engine not available")

    def _tokenize(self, text):
        """Tokenize a text string into input IDs."""
        # Simplified tokenization for demonstration
        return [ord(c) for c in text]

    def _decode(self, ids):
        """Decode token IDs back to text."""
        return "".join(chr(i) for i in ids if 32 <= i < 127)

    def _autoregressive_generate(self, input_ids, max_new_tokens, temperature):
        """Generate tokens autoregressively."""
        # Placeholder: returns dummy tokens
        return input_ids + [ord("A")] * min(max_new_tokens, 10)

    # ---------------------------------------------------------------------------
    # Per-token logprobs and entropy
    # ---------------------------------------------------------------------------

    def _get_per_token_logps_and_entropies(self, input_ids, attention_mask, logits):
        """
        Compute per-token log probabilities and entropies from model logits.

        The log probabilities are computed for the tokens in `input_ids`, and the entropy is
        computed over the full vocabulary distribution at each position. Both are masked using
        `attention_mask` so that padding tokens do not contribute.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, seq_len)`):
                The input token IDs.
            attention_mask (`torch.LongTensor` of shape `(batch_size, seq_len)`):
                The attention mask (1 for real tokens, 0 for padding).
            logits (`torch.FloatTensor` of shape `(batch_size, seq_len, vocab_size)`):
                The model logits.

        Returns:
            `tuple[torch.FloatTensor, torch.FloatTensor]`:
                A tuple of (per_token_logps, per_token_entropies), each of shape
                `(batch_size, seq_len)`.
        """
        # Compute log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)
        per_token_logps = torch.gather(log_probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

        # Compute entropy: H = -sum(p * log(p))
        probs = torch.softmax(logits, dim=-1)
        per_token_entropies = -torch.sum(probs * log_probs, dim=-1)

        # Apply attention mask
        per_token_logps = per_token_logps * attention_mask
        per_token_entropies = per_token_entropies * attention_mask

        return per_token_logps, per_token_entropies

    # ---------------------------------------------------------------------------
    # Input preparation
    # ---------------------------------------------------------------------------

    def _prepare_inputs(self, prompts, completions):
        """
        Prepare model inputs from prompts and completions.

        Tokenizes each prompt-completion pair, creates attention masks, and pads to the same
        length within the batch. Returns a dict with keys `input_ids`, `attention_mask`, and
        `prompt_lengths`.

        Args:
            prompts (`list[str]`):
                The prompt strings.
            completions (`list[str]`):
                The completion strings.

        Returns:
            `dict` with keys:
                - `input_ids` (`list[list[int]]`):
                    Padded input IDs for each sample.
                - `attention_mask` (`list[list[int]]`):
                    Attention mask for each sample.
                - `prompt_lengths` (`list[int]`):
                    The length of the prompt portion in each sample.
        """
        all_input_ids = []
        all_attention_masks = []
        all_prompt_lengths = []

        for prompt, completion in zip(prompts, completions):
            prompt_ids = self._tokenize(prompt)
            completion_ids = self._tokenize(completion)
            input_ids = prompt_ids + completion_ids
            attention_mask = [1] * len(input_ids)
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_prompt_lengths.append(len(prompt_ids))

        # Pad to the same length
        max_len = max(len(ids) for ids in all_input_ids)
        for i in range(len(all_input_ids)):
            pad_len = max_len - len(all_input_ids[i])
            all_input_ids[i] = all_input_ids[i] + [0] * pad_len
            all_attention_masks[i] = all_attention_masks[i] + [0] * pad_len

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "prompt_lengths": all_prompt_lengths,
        }

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------

    def train(self, prompts: list[str]):
        """
        Run the GRPO training loop.

        Args:
            prompts (`list[str]`):
                A list of training prompts.

        Returns:
            `dict`:
                Training metrics.
        """
        for epoch in range(self.num_train_epochs):
            # Step 1: Generate completions
            all_completions = self._generate_single_turn(prompts)

            # Step 2: Flatten prompts and completions for reward computation
            flat_prompts = []
            flat_completions = []
            for prompt, completions in zip(prompts, all_completions):
                for completion in completions:
                    flat_prompts.append(prompt)
                    flat_completions.append(completion)

            # Step 3: Compute rewards
            rewards = self._calculate_rewards(flat_prompts, flat_completions)

            # Step 4: Compute group-level advantages (GRPO-specific)
            advantages = self._compute_grpo_advantages(rewards)

            # Step 5: Prepare inputs
            inputs = self._prepare_inputs(flat_prompts, flat_completions)

            # Step 6: Compute loss and update model
            loss = self._compute_loss(inputs, advantages)

            # Step 7: Log metrics
            mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
            mean_advantage = sum(advantages) / len(advantages) if advantages else 0.0
            self._metrics["train"].append({
                "epoch": epoch,
                "loss": loss,
                "mean_reward": mean_reward,
                "mean_advantage": mean_advantage,
            })

        return {"status": "completed", "epochs": self.num_train_epochs}

    def _compute_grpo_advantages(self, rewards):
        """
        Compute GRPO group-relative advantages.

        Rewards are normalized within each group (of size `num_generations`) to zero mean and
        unit variance.

        Args:
            rewards (`list[float]`):
                Flat list of rewards.

        Returns:
            `list[float]`:
                Normalized advantages.
        """
        advantages = []
        for i in range(0, len(rewards), self.num_generations):
            group = rewards[i : i + self.num_generations]
            mean = sum(group) / len(group)
            std = math.sqrt(sum((r - mean) ** 2 for r in group) / len(group))
            if std < 1e-8:
                advantages.extend([0.0] * len(group))
            else:
                advantages.extend([(r - mean) / std for r in group])
        return advantages

    def _compute_loss(self, inputs, advantages):
        """Compute the GRPO policy loss (placeholder)."""
        # Placeholder: in a real implementation, this would compute the policy gradient loss
        return sum(a ** 2 for a in advantages) / len(advantages) if advantages else 0.0
