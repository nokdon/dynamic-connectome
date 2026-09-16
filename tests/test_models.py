import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dynamic_connectome.artifacts import load_model, load_problem
from dynamic_connectome.config import runtime_config
from dynamic_connectome.stage12.init_params import init_params_b1
from dynamic_connectome.stage12.router import (
    extract_final_graph,
    sample_binary_graph_ste,
)
from dynamic_connectome.stage12.transition import build_transition_operator
from dynamic_connectome.stage12.model_stage2 import forward_stage2
from dynamic_connectome.stage12.sampler import sample_episode_batch
from dynamic_connectome.spiking.spike_surrogate import spike_fn
from dynamic_connectome.spiking.model_b3 import forward_b3_refine
from dynamic_connectome.spiking.loss_b3 import loss_b3_next, teacher_coeffs


def small_config():
    return runtime_config(
        json.loads((Path(__file__).parents[1] / "configs/smoke.json").read_text())[
            "model"
        ]
    )


@pytest.mark.parametrize("budget", [0, 2, 24])
def test_router_respects_support_and_handles_isolated_rows(budget):
    cfg = {**small_config(), "open_budget_topm": budget}
    mask = jnp.asarray(
        [[0, 1, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]], dtype=jnp.float32
    )
    params = init_params_b1(jax.random.PRNGKey(1), mask, cfg)
    out = sample_binary_graph_ste(jax.random.PRNGKey(2), params, mask, cfg)
    z = np.asarray(out["z_hard"])
    assert np.isin(z, [0, 1]).all()
    assert np.all(z <= mask)
    assert z.sum(axis=1).max() <= 2
    assert (z.sum(axis=1) > 0).sum() == min(budget, 3)
    np.testing.assert_array_equal(z, out["z_ste"])
    np.testing.assert_array_equal(z, extract_final_graph(params, mask, cfg)["z_star"])
    assert all(np.isfinite(np.asarray(x)).all() for x in out.values())


def test_router_has_finite_nonzero_surrogate_gradient():
    cfg = {**small_config(), "open_budget_topm": 2}
    mask = jnp.ones((4, 4), dtype=jnp.float32) - jnp.eye(4)
    mask = mask.at[1].set(0)
    params = init_params_b1(jax.random.PRNGKey(3), mask, cfg)
    weights = jnp.arange(16, dtype=jnp.float32).reshape(4, 4)
    grad = jax.grad(
        lambda p: jnp.sum(
            sample_binary_graph_ste(jax.random.PRNGKey(0), p, mask, cfg)["z_ste"]
            * weights
        )
    )(params)
    assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(grad))
    assert np.linalg.norm(grad["S_logits"]) > 0
    assert np.linalg.norm(grad["open_logits"]) > 0
    np.testing.assert_array_equal(
        np.asarray(grad["S_logits"])[np.asarray(mask) == 0], 0
    )


def test_zero_row_transition_and_directed_message_flow():
    cfg = {"eps": 1e-8, "L_msg": 1, "activation": "relu"}
    z = jnp.asarray([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=jnp.float32)
    p = build_transition_operator(z, cfg)["P"]
    np.testing.assert_array_equal(p, z)
    grad = jax.grad(lambda graph: jnp.sum(build_transition_operator(graph, cfg)["P"]))(
        z
    )
    assert np.isfinite(grad).all()
    params = {
        "W_enc": jnp.ones((1, 1)),
        "b_enc": jnp.zeros(1),
        "W_msg": jnp.ones((1, 1, 1)),
        "W_self": jnp.zeros((1, 1, 1)),
        "b_layers": jnp.zeros((1, 1)),
        "W_dec": jnp.ones((1, 1)),
        "b_dec": jnp.zeros(1),
    }
    x = jnp.asarray([[[7.0], [0.0], [0.0]]])
    batch = {"node_inputs": x, "t_idx": jnp.asarray([1])}
    assert float(forward_stage2(params, batch, p, cfg)["pred"][0, 0]) == 7
    batch["t_idx"] = jnp.asarray([2])
    assert float(forward_stage2(params, batch, p, cfg)["pred"][0, 0]) == 0
    batch = {
        "node_inputs": jnp.asarray([[[0.0], [7.0], [0.0]]]),
        "t_idx": jnp.asarray([0]),
    }
    assert float(forward_stage2(params, batch, p, cfg)["pred"][0, 0]) == 0


@pytest.mark.parametrize("model_id", ["learned-0", "hybrid-0"])
def test_predictions_are_batch_independent_and_teacher_free(model_id):
    params, p, cfg, entry = load_model(model_id)
    cfg = {**cfg, "batch_size": 3}
    problem = load_problem()
    batch = sample_episode_batch(
        jax.random.PRNGKey(19), problem["q"], problem["reachable_pairs"], cfg
    )
    single = jax.tree_util.tree_map(lambda value: value[:1], batch)
    if entry["kind"] == "gnn":
        forward = lambda b: forward_stage2(params, b, p, cfg)["pred"]
    else:
        forward = lambda b: forward_b3_refine(
            params, b, p, cfg, entry["spiking_config"]
        )["pred"]
    np.testing.assert_allclose(
        forward(batch)[:1], forward(single), atol=2e-5, rtol=2e-5
    )


def test_spike_forward_is_binary_and_gradient_is_surrogate():
    u = jnp.asarray([-0.1, 0.0, 0.1], dtype=jnp.float32)
    np.testing.assert_array_equal(spike_fn(u, 0.0, {"surrogate_beta": 10.0}), [0, 1, 1])
    grad = jax.grad(lambda v: spike_fn(v, 0.0, {"surrogate_beta": 10.0}).sum())(u)
    assert np.isfinite(grad).all() and np.all(grad > 0)
    assert float(grad[1]) == pytest.approx(2.5)


def test_distillation_requires_explicit_teacher_and_records_nonzero_floor():
    params, p, cfg, entry = load_model("hybrid-0")
    problem = load_problem()
    batch = sample_episode_batch(
        jax.random.PRNGKey(4),
        problem["q"],
        problem["reachable_pairs"],
        {**cfg, "batch_size": 1},
    )
    with pytest.raises(ValueError, match="explicit teacher"):
        loss_b3_next(params, batch, p, cfg, entry["spiking_config"])
    assert teacher_coeffs(100, entry["spiking_config"]) == pytest.approx((0.15, 0.0))


def test_unrestricted_sum_recovers_payload_exactly():
    problem = load_problem()
    cfg = small_config()
    batch = sample_episode_batch(
        jax.random.PRNGKey(17), problem["q"], problem["reachable_pairs"], cfg
    )
    np.testing.assert_array_equal(
        batch["node_inputs"][:, :, : cfg["D"]].sum(axis=1), batch["targets"]
    )
