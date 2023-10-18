from numpy.testing import assert_almost_equal, assert_array_equal

from bandit.env import BanditEnv
from bandit.mab import pure_exploitation


def test_pure_exploitation():
    slippery_bandit_walk = BanditEnv(2, [0.2, 0.8], [1, 1], seed=1019)
    actions, returns, values = pure_exploitation(slippery_bandit_walk, n_episodes=5)

    assert_array_equal(actions, [0, 0, 0, 0, 0])
    assert_array_equal(returns, [1, 0, 0, 0, 0])
    assert_almost_equal(
        values,
        [
            [1.0, 0.0],
            [0.5, 0.0],
            [0.33333333, 0.0],
            [0.25, 0.0],
            [0.2, 0.0],
        ],
    )
