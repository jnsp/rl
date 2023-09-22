from policy_improvement import policy_improvement
from slippery_walk_five import mdp


def test_policy_improvement():
    values = [0, 0.00274725, 0.01098901, 0.03571429, 0.10989011, 0.33241758, 0]
    mdp_slippery_walk_five = mdp

    policy = policy_improvement(values, mdp_slippery_walk_five)
    assert policy == [0, 1, 1, 1, 1, 1, 0]
