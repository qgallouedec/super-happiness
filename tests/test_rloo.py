import unittest

from trl.trainers.rloo_trainer import RLOOTrainer
from trl.utils.rewards import length_reward, correctness_reward


class DummyModel:
    """A minimal dummy model for testing."""
    pass


class TestRLOOTrainer(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel()
        self.prompts = ["What is 2+2?", "Explain gravity."]

    def test_init_defaults(self):
        trainer = RLOOTrainer(model=self.model)
        self.assertEqual(trainer.num_generations, 4)
        self.assertEqual(trainer.max_new_tokens, 256)
        self.assertEqual(trainer.temperature, 0.7)
        self.assertEqual(trainer.kl_coeff, 0.1)
        self.assertEqual(trainer._last_loaded_step, 0)
        self.assertFalse(trainer._vllm_available)

    def test_init_custom(self):
        trainer = RLOOTrainer(
            model=self.model,
            num_generations=8,
            max_new_tokens=128,
            temperature=0.9,
        )
        self.assertEqual(trainer.num_generations, 8)
        self.assertEqual(trainer.max_new_tokens, 128)
        self.assertEqual(trainer.temperature, 0.9)

    def test_generate_single_turn(self):
        trainer = RLOOTrainer(model=self.model, num_generations=2)
        completions = trainer._generate_single_turn(self.prompts)
        self.assertEqual(len(completions), 2)
        for group in completions:
            self.assertEqual(len(group), 2)

    def test_calculate_rewards_no_functions(self):
        trainer = RLOOTrainer(model=self.model)
        rewards = trainer._calculate_rewards(["p1", "p2"], ["c1", "c2"])
        self.assertEqual(rewards, [0.0, 0.0])

    def test_calculate_rewards_with_functions(self):
        trainer = RLOOTrainer(model=self.model, reward_functions=[length_reward])
        rewards = trainer._calculate_rewards(["p1", "p2"], ["short", "a much longer completion"])
        self.assertEqual(len(rewards), 2)
        self.assertGreater(rewards[1], rewards[0])

    def test_calculate_rewards_multiple_functions(self):
        trainer = RLOOTrainer(
            model=self.model,
            reward_functions=[length_reward, correctness_reward],
        )
        rewards = trainer._calculate_rewards(["p"], ["answer is 42"])
        self.assertEqual(len(rewards), 1)
        self.assertGreater(rewards[0], 0.0)

    def test_prepare_inputs(self):
        trainer = RLOOTrainer(model=self.model)
        inputs = trainer._prepare_inputs(["hello", "hi"], ["world", "there!"])
        self.assertIn("input_ids", inputs)
        self.assertIn("attention_mask", inputs)
        self.assertIn("prompt_lengths", inputs)
        lengths = [len(ids) for ids in inputs["input_ids"]]
        self.assertEqual(len(set(lengths)), 1)

    def test_compute_rloo_advantages(self):
        trainer = RLOOTrainer(model=self.model, num_generations=4)
        rewards = [1.0, 2.0, 3.0, 4.0]
        advantages = trainer._compute_rloo_advantages(rewards)
        self.assertEqual(len(advantages), 4)
        # With leave-one-out, advantages should sum to zero within a group
        self.assertAlmostEqual(sum(advantages), 0.0, places=5)

    def test_compute_rloo_advantages_single(self):
        trainer = RLOOTrainer(model=self.model, num_generations=1)
        rewards = [3.0, 5.0]
        advantages = trainer._compute_rloo_advantages(rewards)
        # With only 1 generation, baseline is 0, so advantage = reward
        self.assertEqual(advantages, [3.0, 5.0])

    def test_train(self):
        trainer = RLOOTrainer(
            model=self.model,
            reward_functions=[length_reward],
            num_generations=2,
            num_train_epochs=2,
        )
        result = trainer.train(self.prompts)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["epochs"], 2)
        self.assertEqual(len(trainer._metrics["train"]), 2)

    def test_model_card(self):
        trainer = RLOOTrainer(model=self.model, args={"lr": 1e-6})
        card = trainer.generate_model_card()
        self.assertIn("RLOOTrainer", card)
        self.assertIn("lr", card)


if __name__ == "__main__":
    unittest.main()
