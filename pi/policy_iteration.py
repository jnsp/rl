import numpy as np

from pi.mdp import MDP


def value_iteration(
    mdp: MDP, gamma: float = 1.0, epsilon: float = 1e-10
) -> tuple[np.ndarray, list[int]]:
    prev_state_values = mdp.zero_state_values()

    while True:
        action_values = mdp.zero_action_values()

        for state in mdp.state_space:
            for action in mdp.action_space:
                action_values[state][action] = value(
                    mdp, state, action, prev_state_values, gamma
                )

        state_values = np.max(action_values, axis=1)

        if converged(prev_state_values, state_values, epsilon):
            break

        prev_state_values = state_values

    policy = list(np.argmax(action_values, axis=1))

    return state_values, policy


def policy_iteration(
    mdp: MDP, gamma: float = 1.0, epsilon: float = 1e-10
) -> tuple[np.ndarray, list[int]]:
    state_values = mdp.zero_state_values()
    policy = mdp.random_policy()

    while True:
        state_values = policy_evaluation(policy, mdp, gamma, epsilon)
        new_policy = policy_improvement(state_values, mdp, gamma)

        if policy == new_policy:
            break

        policy = new_policy

    return state_values, new_policy


def policy_evaluation(
    policy: list[int], mdp: MDP, gamma: float = 1.0, epsilon: float = 1e-10
) -> np.ndarray:
    prev_state_values = mdp.zero_state_values()

    while True:
        state_values = mdp.zero_state_values()

        for state in mdp.state_space:
            action = policy[state]
            state_values[state] = value(mdp, state, action, prev_state_values, gamma)

        if converged(prev_state_values, state_values, epsilon):
            break

        prev_state_values = state_values

    return state_values


def policy_improvement(state_values: np.ndarray, mdp: MDP, gamma=1.0) -> list[int]:
    action_values = mdp.zero_action_values()

    for state in mdp.state_space:
        for action in mdp.action_space:
            action_values[state][action] = value(
                mdp, state, action, state_values, gamma
            )

    policy = list(np.argmax(action_values, axis=1))
    return policy


def value(mdp: MDP, state: int, action: int, values: np.ndarray, gamma: float):
    value = 0
    for transition_probability, next_state, reward, is_terminal in mdp[state][action]:
        discounted_value_of_next_state = gamma * values[next_state] * (not is_terminal)
        return_value = reward + discounted_value_of_next_state
        value += transition_probability * return_value
    return value


def converged(previous: np.ndarray, current: np.ndarray, epsilon: float) -> bool:
    return np.max(np.abs(previous - current)) < epsilon
