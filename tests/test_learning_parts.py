import numpy as np
import torch

from airfed.config import VARIANTS, Config
from airfed.env.uav_env import UAVEdgeEnv
from airfed.fl.federation import Federation
from airfed.fl.quantize import bit_widths, compress, full_precision
from airfed.fl.reputation import Reputation
from airfed.models.agent import Agent, offload_terms, sample_velocity
from airfed.train import to_torch


def test_bit_widths_follow_gradient_rank():
    bits = bit_widths(torch.tensor([0.1, 5.0, 1.0, 0.0]), 4, 16)
    assert bits.tolist() == [8, 16, 12, 4]


def test_quantization_error_shrinks_with_bits():
    torch.manual_seed(0)
    t = torch.randn(1000)
    low, _ = compress([t], [torch.zeros(1000)], 4, 4)
    high, _ = compress([t], [torch.zeros(1000)], 16, 16)
    assert (high[0] - t).abs().max() < (low[0] - t).abs().max()
    assert (high[0] - t).abs().max() < 1e-3


def test_quantized_message_is_smaller_than_float32():
    t = torch.randn(5000)
    _, q_bits = compress([t], [torch.rand(5000)], 4, 16)
    _, f_bits = full_precision([t])
    assert q_bits < 0.5 * f_bits


def test_reputation_drops_for_unreliable_partner():
    rep = Reputation(3)
    for _ in range(10):
        rep.record_exchange(0, True)
        rep.record_exchange(1, False)
        rep.update(np.array([10, 10, 0]), np.array([10, 2, 0]))
    assert rep.score[0] == rep.score[2] == 1.0       # reliable, or simply no evidence yet
    assert rep.score[1] < 0.6                         # misses deadlines and drops exchanges


def test_agent_outputs_have_expected_shapes():
    cfg = Config()
    env = UAVEdgeEnv(cfg.env, seed=0)
    obs = to_torch(env.reset())
    for use_gat in (True, False):
        agent = Agent(env.K, cfg.model, use_gat)
        mean, log_std, logits, value, hidden = agent(obs, 0, torch.zeros(cfg.model.gru_hidden))
        assert mean.shape == log_std.shape == (2,)
        assert logits.shape == (env.K,)
        assert value.shape == ()
        assert hidden.shape == (cfg.model.gru_hidden,)
        # masked slots get zero probability
        probs = torch.softmax(logits, -1)
        assert torch.all(probs[~obs["slot_mask"][0]] == 0)


def test_velocity_sample_is_bounded():
    v, logp, ent = sample_velocity(torch.zeros(2), torch.zeros(2), 20.0)
    assert torch.all(v.abs() <= 20.0) and torch.isfinite(logp)


def test_offload_log_prob_ignores_masked_slots():
    logits = torch.tensor([0.0, 0.0, float("-inf")])
    logp, ent, _ = offload_terms(logits, torch.tensor([3.0, 1.0, 0.0]))
    assert torch.isfinite(logp) and torch.isfinite(ent)


def test_federated_round_averages_neighbours():
    cfg = Config()
    env = UAVEdgeEnv(cfg.env, seed=0)
    env.reset()
    agents = [Agent(env.K, cfg.model, True) for _ in range(env.K)]
    for a in agents:
        a.last_grad = {}
    fed = Federation(env.K, cfg.fed, VARIANTS["no_quant"], seed=0)
    before = agents[0].critic[0].weight.detach().clone()
    fed.step(agents, env, elapsed=1000.0)          # every timer fires
    assert fed.rounds == env.K
    assert fed.sent_bits.sum() > 0
    has_partner = any(env.d_uu[0, j] <= cfg.env.r_comm for j in range(1, env.K))
    changed = not torch.equal(before, agents[0].critic[0].weight)
    assert changed == has_partner
