import numpy as np


def policy_evaluation(policy, mdp, gamma=1.0, epsilon=1e-10):
    state_space = mdp.keys()
    prev_state_values = np.zeros(len(state_space))

    while True:
        state_values = np.zeros(len(state_space))

        for state in state_space:
            action = policy[state]
            transitions = mdp[state][action]
            state_values[state] = state_value(transitions, prev_state_values, gamma)

        if converged(prev_state_values, state_values, epsilon):
            break

        prev_state_values = state_values

    return state_values


def state_value(transitions, prev_values, gamma):
    value = 0

    for prob, next_state, reward, done in transitions:
        discounted_value_of_next_state = gamma * prev_values[next_state] * (not done)
        return_value = reward + discounted_value_of_next_state
        value += prob * return_value

    return value


def converged(previous, current, epsilon):
    return np.max(np.abs(previous - current)) < epsilon
