import pytest

from bandit.env import BanditEnv


def test_validity_check_number_of_arms():
    with pytest.raises(ValueError, match="n_arms"):
        BanditEnv(2, [1, 0, 0], [1, 1, 1])

    with pytest.raises(ValueError, match="n_arms"):
        BanditEnv(2, [1, 0, 0], [1, 1])


def test_valid_pay_probs():
    with pytest.raises(ValueError, match="pay_dist"):
        BanditEnv(2, [-1, 0], [1, 1])


def test_valid_reward_params():
    with pytest.raises(ValueError, match="Sigma must be non-negative"):
        BanditEnv(2, [1, 0], [(1, -1), (1, -1)])


def test_simple_bandit():
    bandit = BanditEnv(2, [1, 0], [1, 1])
    rewards = [bandit.step(0) for _ in range(100)]
    assert all(reward == 1 for reward in rewards)


def test_prob_bandit():
    bandit = BanditEnv(2, [0.5, 0], [1, 1])
    rewards = [bandit.step(0) for _ in range(10000)]
    assert sum(rewards) == pytest.approx(5000, rel=0.05)


def test_get_n_arms():
    bandit = BanditEnv(2, [1, 0], [1, 1])
    assert bandit.n_arms == 2


def test_reward_from_normal_dist():
    bandit = BanditEnv(2, [1, 0], [(1, 1), (1, 1)], seed=1012)
    rewards = [bandit.step(0) for _ in range(5)]
    assert rewards == [
        0.2304217607689154,
        1.0710329360509505,
        0.758468638735911,
        0.6379031512058656,
        0.9571375785283169,
    ]
