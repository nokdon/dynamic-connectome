import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dynamic_connectome.artifacts import load_model, load_problem
from dynamic_connectome.evaluation import batches_for_key, prediction_function, score
from dynamic_connectome.training import fit, make_update, train


def test_best_checkpoint_is_the_state_actually_validated():
    # Training target and validation target disagree: best is an intermediate state.
    opt, step = make_update(lambda p, b: (p - b) ** 2, learning_rate=0.4, clip=100)
    best, report = fit(
        jnp.asarray(0.0),
        step,
        opt,
        lambda _: jnp.asarray(3.0),
        lambda p: float((p - 0.8) ** 2),
        steps=8,
        eval_every=1,
    )
    assert 0 < report["selected_step"] < 8
    assert float((best - 0.8) ** 2) == pytest.approx(report["validation_mse"])
    assert report["validation_mse"] == min(
        r["validation_mse"] for r in report["history"]
    )


def test_full_smoke_saves_exact_selected_scores_and_frozen_graph(tmp_path):
    config = Path(__file__).parents[1] / "configs/smoke.json"
    output = tmp_path / "run"
    report = train(config, output, progress=lambda _: None)
    spec = json.loads((output / "config.json").read_text())
    schedule = spec["schedule"]
    problem = load_problem(output)
    graph = None
    for model_id, phase in [("stage2", "stage2"), ("hybrid", "hybrid")]:
        params, p, cfg, entry = load_model(model_id, output)
        if graph is None:
            graph = np.asarray(p)
        else:
            np.testing.assert_array_equal(graph, p)
        cfg = {**cfg, "batch_size": schedule["eval_batch_size"]}
        predict = prediction_function(params, p, cfg, entry["spiking_config"])
        for stream, field, count in [
            ("validation", "validation_mse", schedule["validation_batches"]),
            ("test", "test", schedule["test_batches"]),
        ]:
            key = jax.random.fold_in(
                jax.random.PRNGKey(spec["seed"]), report["stream_ids"][stream]
            )
            actual = score(predict, batches_for_key(key, count, problem, cfg))["mse"]
            expected = (
                report[phase][field]
                if stream == "validation"
                else report[phase][field]["mse"]
            )
            assert actual == pytest.approx(expected, abs=2e-6)
    with pytest.raises(FileExistsError):
        train(config, output, progress=lambda _: None)
    assert len(set(report["stream_ids"].values())) == len(report["stream_ids"])
