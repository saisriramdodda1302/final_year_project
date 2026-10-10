import numpy as np
import pytest

from airfed.config import Config
from airfed.env import channel as ch
from airfed.env.uav_env import UAVEdgeEnv


@pytest.fixture
def env():
    e = UAVEdgeEnv(Config().env, seed=1)
    e.reset()
    return e


def test_db_helpers_round_trip():
    assert ch.db_to_linear(-30) == pytest.approx(1e-3)
    assert ch.watt_to_dbm(ch.dbm_to_watt(-90.0)) == pytest.approx(-90.0)


def test_flight_power_matches_eq19():
    cfg = Config().env
    assert ch.flight_power(cfg, 0.0) == pytest.approx(80.0)
    assert ch.flight_power(cfg, 20.0) == pytest.approx(80.0 + 0.5 * 1.225 * 0.1 * 0.3 * 8000)


def test_coverage_footprint_is_a_few_hundred_metres(env):
    covered = env.d_ug[env.cover]
    assert 250 < covered.max() < 350


def test_time_step_respects_eq38():
    cfg = Config().env
    assert cfg.dt >= 2 * cfg.v_max / cfg.accel


def test_speed_limit_and_area_are_enforced(env):
    obs, _, _, _ = env.step(np.full((env.K, 2), 100.0), np.eye(env.S + 1)[[0] * env.K])
    assert np.all(np.linalg.norm(env.vel, axis=1) <= env.cfg.v_max + 1e-6)
    assert np.all((env.pos[:, :2] >= 0) & (env.pos[:, :2] <= env.cfg.area))


def test_hovering_costs_hover_power(env):
    _, _, _, info = env.step(np.zeros((env.K, 2)), np.eye(env.S + 1)[[0] * env.K])
    assert np.all(info["step_energy"] >= env.cfg.p_hover * env.cfg.dt)


def test_episode_length_and_metrics(env):
    done, steps = False, 0
    while not done:
        _, rewards, done, _ = env.step(np.zeros((env.K, 2)), np.eye(env.S + 1)[[0] * env.K])
        assert rewards.shape == (env.K,)
        steps += 1
    assert steps == env.cfg.steps == 30
    m = env.episode_metrics()
    assert 0 < m["time"] < 5
    assert 0.0 <= m["coverage"] <= 1.0
    assert m["tasks"] > 0


def test_offloading_only_targets_reachable_neighbours(env):
    obs = env._observe()
    probs = np.zeros((env.K, env.S + 1))
    probs[:, 1] = 1.0                                   # try to forward everything to the nearest slot
    probs[~obs["slot_mask"][:, 1], 1] = 0.0
    probs[~obs["slot_mask"][:, 1], 0] = 1.0             # UAVs without a neighbour stay local
    before = env.exec_total.copy()
    env.step(np.zeros((env.K, 2)), probs)
    assert env.exec_total.sum() > before.sum()


def test_slot_mask_always_allows_local(env):
    obs = env._observe()
    assert obs["slot_mask"][:, 0].all()


def test_pushing_into_the_border_costs_energy(env):
    # park every UAV on the left edge, then compare hovering with pushing further left
    env.pos[:, 0] = 0.0
    local = np.eye(env.S + 1)[[0] * env.K]
    _, _, _, hover = env.step(np.zeros((env.K, 2)), local)
    env.pos[:, 0] = 0.0
    _, _, _, push = env.step(np.tile([-15.0, 0.0], (env.K, 1)), local)
    assert np.all(env.pos[:, 0] == 0.0)                       # the geofence held them
    assert np.all(push["step_energy"] > hover["step_energy"])  # but trying still cost power
