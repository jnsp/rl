import numpy as np

from policy_evaluation import policy_evaluation
from slippery_walk_five import mdp


def test_policy_evaluation_left_policy():
    left_policy = [0, 0, 0, 0, 0, 0, 0]
    mdp_slippery_walk_five = mdp

    value_func = policy_evaluation(left_policy, mdp_slippery_walk_five)
    expected = [0, 0.00274725, 0.01098901, 0.03571429, 0.10989011, 0.33241758, 0]

    np.testing.assert_array_almost_equal(value_func, expected)


def test_policy_evaluation_right_policy():
    right_policy = [1, 1, 1, 1, 1, 1, 1]
    mdp_slippery_walk_five = mdp

    value_func = policy_evaluation(right_policy, mdp_slippery_walk_five)
    expected = [0, 0.66758242, 0.89010989, 0.96428571, 0.98901099, 0.9972527, 0]

    np.testing.assert_array_almost_equal(value_func, expected)
