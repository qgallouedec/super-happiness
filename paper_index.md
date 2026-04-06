# Paper Index

This file tracks the research papers implemented in this repository.

## GRPO — Group-Relative Policy Optimization

- **Paper**: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- **Trainer**: `trl/trainers/grpo_trainer.py`
- **Summary**: GRPO eliminates the need for a critic model by estimating the baseline from group scores. For each prompt, multiple completions are generated and their rewards are normalized within the group to compute advantages.

## RLOO — REINFORCE Leave-One-Out

- **Paper**: [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740)
- **Trainer**: `trl/trainers/rloo_trainer.py`
- **Summary**: RLOO uses a leave-one-out baseline estimator for variance reduction. For each completion, the baseline is the average reward of all other completions in the same group, providing an unbiased and low-variance gradient estimate.
