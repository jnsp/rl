import numpy as np


def policy_improvement(values, mdp, gamma=1.0):
    state_space = mdp.keys()
    action_space = mdp[0].keys()
    action_values = np.zeros((len(state_space), len(action_space)))

    for state in state_space:
        for action in action_space:
            transitions = mdp[state][action]
            action_values[state][action] = action_value(transitions, values, gamma)

    policy = list(np.argmax(action_values, axis=1))
    return policy


def action_value(transitions, values, gamma):
    value = 0

    for prob, next_state, reward, done in transitions:
        discounted_value_of_next_state = gamma * values[next_state] * (not done)
        return_value = reward + discounted_value_of_next_state
        value += prob * return_value

    return value
