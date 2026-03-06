def compute_reward_from_functions(reward_functions, prompts, completions):
    """
    Compute total rewards by summing results from multiple reward functions.

    Args:
        reward_functions (`list[callable]`):
            A list of reward functions. Each function takes (prompts, completions) and returns
            a list of floats.
        prompts (`list[str]`):
            The prompt strings.
        completions (`list[str]`):
            The completion strings.

    Returns:
        `list[float]`:
            A list of total rewards, one per completion.
    """
    total_rewards = [0.0] * len(completions)
    for reward_fn in reward_functions:
        rewards = reward_fn(prompts, completions)
        for i in range(len(completions)):
            total_rewards[i] += rewards[i]
    return total_rewards


def length_reward(prompts, completions):
    """
    A simple reward function that rewards longer completions.

    Args:
        prompts (`list[str]`):
            The prompt strings (unused but required by interface).
        completions (`list[str]`):
            The completion strings.

    Returns:
        `list[float]`:
            Reward proportional to completion length.
    """
    return [len(c) / 100.0 for c in completions]


def correctness_reward(prompts, completions):
    """
    A simple reward function that checks if the completion contains the expected answer.

    Args:
        prompts (`list[str]`):
            The prompt strings.
        completions (`list[str]`):
            The completion strings.

    Returns:
        `list[float]`:
            1.0 if the completion contains a number, 0.0 otherwise.
    """
    rewards = []
    for completion in completions:
        has_number = any(c.isdigit() for c in completion)
        rewards.append(1.0 if has_number else 0.0)
    return rewards
