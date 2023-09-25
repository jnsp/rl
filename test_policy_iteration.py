import numpy as np

from mdp import MDP
from policy_iteration import (
    policy_evaluation,
    policy_improvement,
    policy_iteration,
    value_iteration,
)
from transitions import frozen_lake_gridworld, slippery_walk_five


def test_policy_evaluation_left_policy():
    left_policy = [0, 0, 0, 0, 0, 0, 0]
    mdp = MDP(slippery_walk_five)

    value_func = policy_evaluation(left_policy, mdp)
    expected = [0, 0.00274725, 0.01098901, 0.03571429, 0.10989011, 0.33241758, 0]

    np.testing.assert_array_almost_equal(value_func, expected)


def test_policy_evaluation_right_policy():
    right_policy = [1, 1, 1, 1, 1, 1, 1]
    mdp = MDP(slippery_walk_five)

    value_func = policy_evaluation(right_policy, mdp)
    expected = [0, 0.66758242, 0.89010989, 0.96428571, 0.98901099, 0.9972527, 0]

    np.testing.assert_array_almost_equal(value_func, expected)


def test_policy_improvement():
    values = np.array(
        [0, 0.00274725, 0.01098901, 0.03571429, 0.10989011, 0.33241758, 0]
    )
    mdp = MDP(slippery_walk_five)

    policy = policy_improvement(values, mdp)
    assert policy == [0, 1, 1, 1, 1, 1, 0]


def test_policy_iteration():
    mdp = MDP(slippery_walk_five)

    state_values, policy = policy_iteration(mdp)
    expected_policy = [0, 1, 1, 1, 1, 1, 0]
    expected = [0, 0.66758242, 0.89010989, 0.96428571, 0.98901099, 0.9972527, 0]

    assert policy == expected_policy
    np.testing.assert_array_almost_equal(state_values, expected)


def test_policy_iteration_frozen_lake_gridworld():
    mdp = MDP(frozen_lake_gridworld)
    gamma = 0.99

    state_values, policy = policy_iteration(mdp, gamma)
    expected_policy = [0, 3, 3, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0]
    expected = [
        0.54202593, 0.49880319, 0.47069569, 0.4568517,
        0.55845096, 0.0, 0.35834807, 0.0,
        0.59179874, 0.64307982, 0.61520756, 0.0,
        0.0, 0.74172044, 0.86283743, 0.0,
    ]  # fmt: skip

    assert policy == expected_policy
    np.testing.assert_array_almost_equal(state_values, expected)


def test_value_iteration():
    mdp = MDP(slippery_walk_five)

    state_values, policy = value_iteration(mdp)
    expected_policy = [0, 1, 1, 1, 1, 1, 0]
    expected = [0, 0.66758242, 0.89010989, 0.96428571, 0.98901099, 0.9972527, 0]

    assert policy == expected_policy
    np.testing.assert_array_almost_equal(state_values, expected)
