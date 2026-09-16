import json
from pathlib import Path

import jax
import numpy as np
import pytest

from dynamic_connectome.artifacts import load_model, load_problem
from dynamic_connectome.baselines import (
    sample_random_sparse_matched_row_degree,
    build_heuristic_shortest_path_budgeted_graph,
)
from dynamic_connectome.evaluation import batches_for_key, prediction_function, score
from dynamic_connectome.train_baselines import train_baselines


def test_graph_builders_recover_the_released_controls():
    problem = load_problem()
    _, anchor, _, _ = load_model("learned-0")
    degrees = (np.asarray(anchor) > 0).sum(axis=1)
    for seed in range(5):
        graph = sample_random_sparse_matched_row_degree(
            problem["M"], degrees, 200000 + seed
        )
        _, actual, _, _ = load_model(f"random-{seed}")
        np.testing.assert_array_equal(graph, np.asarray(actual) > 0)
        np.testing.assert_array_equal(graph.sum(axis=1), degrees)
    q = np.asarray(problem["q"])
    problem["L"] = np.linalg.norm(q[:, None, :] - q[None, :, :], axis=-1)
    graph = build_heuristic_shortest_path_budgeted_graph(
        problem, {"k_router": 2, "open_budget_topm": 22}
    )
    _, actual, _, _ = load_model("heuristic-0")
    np.testing.assert_array_equal(graph, np.asarray(actual) > 0)


def test_baseline_training_saves_validation_selected_weights(tmp_path):
    config = Path(__file__).parents[1] / "configs/smoke.json"
    output = tmp_path / "controls"
    report = train_baselines(config, output, progress=lambda _: None)
    spec = json.loads(config.read_text())
    problem = load_problem(output)
    for family in ("dense", "random", "heuristic", "dense_soft"):
        params, p, cfg, entry = load_model(f"{family}-0", output)
        cfg = {**cfg, "batch_size": spec["schedule"]["eval_batch_size"]}
        key = jax.random.fold_in(jax.random.PRNGKey(spec["seed"]), 100)
        batches = batches_for_key(
            key, spec["schedule"]["validation_batches"], problem, cfg
        )
        actual = score(prediction_function(params, p, cfg), batches)["mse"]
        assert actual == pytest.approx(
            report["families"][family]["validation_mse"], abs=2e-6
        )
