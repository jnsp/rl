from mdp import MDP
from transitions import slippery_walk_five


def test_mdp():
    mdp = MDP(slippery_walk_five)
    assert mdp.state_space == [0, 1, 2, 3, 4, 5, 6]
    assert mdp.action_space == [0, 1]
    assert mdp.size_state_space == 7
    assert mdp.size_action_space == 2
    assert mdp[1][0] == [
        (1 / 2, 0, 0, True),
        (1 / 3, 1, 0, False),
        (1 / 6, 2, 0, False),
    ]
    assert mdp.zero_state_values().shape == (7,)
    assert mdp.zero_action_values().shape == (7, 2)
    assert len(mdp.random_policy()) == 7
