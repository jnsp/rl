import random
from collections.abc import Iterable


class BanditEnv:
    def __init__(
        self,
        n_arms: int,
        pay_probs: list[float],
        rewards: list[float] | list[tuple[float, float]],
        seed: int | None = None,
    ):
        self._check_validity_number_of_arms(n_arms, pay_probs, rewards)
        self._check_validity_pay_probs(pay_probs)

        self._n_arms = n_arms
        self._pay_dist = pay_probs
        self._rewards = rewards
        self._rng = random.Random(seed)

    def _check_validity_number_of_arms(self, n_arms, pay_probs, reward_dist):
        n_pay_dist = len(pay_probs)
        n_reward_dist = len(reward_dist)

        if not (n_arms == n_pay_dist == n_reward_dist):
            raise ValueError(
                f"n_arms ({n_arms}), length of pay_probs ({n_pay_dist}), "
                "and length of reward_dist ({n_reward_dist}) must be equal"
            )

    def _check_validity_pay_probs(self, pay_probs):
        if not all(0 <= p <= 1 for p in pay_probs):
            raise ValueError("pay_dist must be a list of floats between 0 and 1")

    def step(self, action: int):
        reward_params = self._rewards[action] if self._does_pay(action) else 0

        if not isinstance(reward_params, Iterable):
            reward_params = (reward_params, 0)

        reward = self._rng.normalvariate(*reward_params)
        return reward

    def _does_pay(self, action: int) -> bool:
        return self._rng.random() < self._pay_dist[action]
