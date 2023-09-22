import random

import numpy as np


class MDP:
    def __init__(
        self, trasitions: dict[int, dict[int, list[tuple[float, int, float, bool]]]]
    ):
        self._transitions = trasitions

    def __getitem__(
        self, state: int
    ) -> dict[int, list[tuple[float, int, float, bool]]]:
        return self._transitions[state]

    @property
    def state_space(self) -> list[int]:
        return list(self._transitions.keys())

    @property
    def size_state_space(self) -> int:
        return len(self.state_space)

    @property
    def action_space(self) -> list[int]:
        return list(self._transitions[0].keys())

    @property
    def size_action_space(self) -> int:
        return len(self.action_space)

    def zero_state_values(self) -> np.ndarray:
        return np.zeros(self.size_state_space)

    def zero_action_values(self) -> np.ndarray:
        return np.zeros((self.size_state_space, self.size_action_space))

    def random_policy(self) -> list[int]:
        return random.choices(self.action_space, k=self.size_state_space)
