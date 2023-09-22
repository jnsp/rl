import numpy as np

from mdp import MDP


def value_iteration(
    mdp: MDP, gamma: float = 1.0, epsilon: float = 1e-10
) -> tuple[np.ndarray, list[int]]:
    state_values = mdp.zero_state_values()

    while True:
        action_values = mdp.zero_action_values()

        for state in mdp.state_space:
            for action in mdp.action_space:
                transitions = mdp[state][action]
                action_values[state][action] = value(transitions, state_values, gamma)

        new_state_values = np.max(action_values, axis=1)

        if converged(state_values, new_state_values, epsilon):
            break

        state_values = new_state_values

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
            transitions = mdp[state][action]
            state_values[state] = value(transitions, prev_state_values, gamma)

        if converged(prev_state_values, state_values, epsilon):
            break

        prev_state_values = state_values

    return state_values


def policy_improvement(state_values: np.ndarray, mdp: MDP, gamma=1.0) -> list[int]:
    action_values = mdp.zero_action_values()

    for state in mdp.state_space:
        for action in mdp.action_space:
            transitions = mdp[state][action]
            action_values[state][action] = value(transitions, state_values, gamma)

    policy = list(np.argmax(action_values, axis=1))
    return policy


def value(
    transitions: list[tuple[float, int, float, bool]], values: np.ndarray, gamma: float
) -> float:
    value = 0

    for prob, next_state, reward, done in transitions:
        discounted_value_of_next_state = gamma * values[next_state] * (not done)
        return_value = reward + discounted_value_of_next_state
        value += prob * return_value

    return value


def converged(previous: np.ndarray, current: np.ndarray, epsilon: float) -> bool:
    return np.max(np.abs(previous - current)) < epsilon
