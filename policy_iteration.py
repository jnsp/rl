import random

import numpy as np


def policy_iteration(mdp, gamma=1.0, epsilon=1e-10):
    state_space = mdp.keys()
    action_space = list(mdp[0].keys())
    state_values = np.zeros(len(state_space))
    policy = random.choices(action_space, k=len(state_space))

    while True:
        state_values = policy_evaluation(policy, mdp, gamma, epsilon)
        new_policy = policy_improvement(state_values, mdp, gamma)

        if policy == new_policy:
            break

        policy = new_policy

    return state_values, new_policy


def policy_evaluation(policy, mdp, gamma=1.0, epsilon=1e-10):
    state_space = mdp.keys()
    prev_state_values = np.zeros(len(state_space))

    while True:
        state_values = np.zeros(len(state_space))

        for state in state_space:
            action = policy[state]
            transitions = mdp[state][action]
            state_values[state] = value(transitions, prev_state_values, gamma)

        if converged(prev_state_values, state_values, epsilon):
            break

        prev_state_values = state_values

    return state_values


def policy_improvement(state_values, mdp, gamma=1.0):
    state_space = mdp.keys()
    action_space = mdp[0].keys()
    action_values = np.zeros((len(state_space), len(action_space)))

    for state in state_space:
        for action in action_space:
            transitions = mdp[state][action]
            action_values[state][action] = value(transitions, state_values, gamma)

    policy = list(np.argmax(action_values, axis=1))
    return policy


def value(transitions, values, gamma):
    value = 0

    for prob, next_state, reward, done in transitions:
        discounted_value_of_next_state = gamma * values[next_state] * (not done)
        return_value = reward + discounted_value_of_next_state
        value += prob * return_value

    return value


def converged(previous, current, epsilon):
    return np.max(np.abs(previous - current)) < epsilon
