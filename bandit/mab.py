import random

import numpy as np

from bandit.env import BanditEnv


def pure_exploitation(env: BanditEnv, n_episodes: int = 5000) -> tuple[np.ndarray, ...]:
    action_values = np.zeros(env.n_arms)
    n_trials = np.zeros(env.n_arms)
    action_values_episodes = np.empty((n_episodes, env.n_arms))
    returns_episodes = np.empty(n_episodes)
    actions_episodes = np.empty(n_episodes, dtype=int)

    for e in range(n_episodes):
        action = int(np.argmax(action_values))
        reward = env.step(action)

        n_trials[action] += 1
        action_values[action] += (reward - action_values[action]) / n_trials[action]

        actions_episodes[e] = action
        returns_episodes[e] = reward
        action_values_episodes[e] = action_values

    return actions_episodes, returns_episodes, action_values_episodes


def pure_exploration(
    env: BanditEnv, n_episodes: int = 5000, seed: int | None = None
) -> tuple[np.ndarray, ...]:
    action_values = np.zeros(env.n_arms)
    n_trials = np.zeros(env.n_arms)
    action_values_episodes = np.empty((n_episodes, env.n_arms))
    returns_episodes = np.empty(n_episodes)
    actions_episodes = np.empty(n_episodes, dtype=int)
    rng = np.random.default_rng(seed)

    for e in range(n_episodes):
        action = rng.integers(0, env.n_arms)
        reward = env.step(action)

        n_trials[action] += 1
        action_values[action] += (reward - action_values[action]) / n_trials[action]

        actions_episodes[e] = action
        returns_episodes[e] = reward
        action_values_episodes[e] = action_values

    return actions_episodes, returns_episodes, action_values_episodes
