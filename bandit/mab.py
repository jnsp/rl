import numpy as np


def pure_exploitation(env, n_episodes=5000):
    action_values = np.zeros(env.n_arms)
    n_trials = np.zeros(env.n_arms)
    action_values_episodes = np.empty((n_episodes, env.n_arms))
    returns_episodes = np.empty(n_episodes)
    actions_episodes = np.empty(n_episodes, dtype=int)

    for e in range(n_episodes):
        action = np.argmax(action_values)
        reward = env.step(action)

        n_trials[action] += 1
        action_values[action] += (reward - action_values[action]) / n_trials[action]

        actions_episodes[e] = action
        returns_episodes[e] = reward
        action_values_episodes[e] = action_values

    return actions_episodes, returns_episodes, action_values_episodes