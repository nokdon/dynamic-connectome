"""New baseline training with the same held-out selection protocol as the core."""

import json
from pathlib import Path
import shutil

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import (
    ARTIFACT_ROOT,
    load_model,
    load_problem,
    read_catalog,
    save_model,
    sha256,
)
from .baselines import (
    build_heuristic_shortest_path_budgeted_graph,
    sample_random_sparse_matched_row_degree,
)
from .config import load_training_config, runtime_config
from .evaluation import batches_for_key, score
from .stage12.init_params import init_params_b1
from .stage12.model_stage2 import forward_stage2, init_params_stage2_from_stage1
from .stage12.loss_stage2 import loss_stage2
from .stage12.sampler import sample_episode_batch
from .stage12.transition import build_transition_operator
from .training import fit, make_update


def train_baselines(config_path, output, reference=ARTIFACT_ROOT, *, progress=print):
    spec = load_training_config(config_path)
    cfg = runtime_config(spec["model"])
    schedule = spec["schedule"]
    reference_catalog = read_catalog(reference)
    anchor = next(m for m in reference_catalog["models"] if m["family"] == "learned")
    _, anchor_p, _, _ = load_model(anchor["id"], reference, reference_catalog)
    problem = load_problem(reference, reference_catalog)
    q = np.asarray(problem["q"])
    problem_with_lengths = {
        **problem,
        "L": np.linalg.norm(q[:, None, :] - q[None, :, :], axis=-1),
    }
    graphs = {
        "dense": np.asarray(problem["M"]),
        "random": sample_random_sparse_matched_row_degree(
            problem["M"], (np.asarray(anchor_p) > 0).sum(axis=1), 200000 + spec["seed"]
        ),
        "heuristic": build_heuristic_shortest_path_budgeted_graph(
            problem_with_lengths, cfg
        ),
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(
        Path(reference) / reference_catalog["problem"]["path"], output / "problem.npz"
    )
    (output / "config.json").write_text(json.dumps(spec, indent=2) + "\n")
    key = jax.random.PRNGKey(spec["seed"])
    init_key = jax.random.fold_in(key, 0)
    train_key = jax.random.fold_in(key, 50)
    val_key = jax.random.fold_in(key, 100)
    test_key = jax.random.fold_in(key, 200)
    eval_cfg = {**cfg, "batch_size": schedule["eval_batch_size"]}
    validation = batches_for_key(
        val_key, schedule["validation_batches"], problem, eval_cfg
    )
    theta = init_params_stage2_from_stage1(init_params_b1(init_key, problem["M"], cfg))
    draw = lambda step: sample_episode_batch(
        jax.random.fold_in(train_key, step),
        problem["q"],
        problem["reachable_pairs"],
        cfg,
    )
    steps = schedule["stage1_steps"] + schedule["stage2_steps"]
    entries = []
    selected = []
    report = {
        "protocol": "New baseline training; common initialization, training and validation draws; post-update selection; common test evaluated after all selections.",
        "reference_graph_id": anchor["id"],
        "reference_graph_artifact_sha256": anchor["sha256"],
        "random_graph_seed": 200000 + spec["seed"],
        "random_degree_template": "reference learned graph",
        "stream_ids": {
            "initialization": 0,
            "training": 50,
            "validation": 100,
            "test": 200,
        },
        "families": {},
    }
    for family in ("dense", "random", "heuristic", "dense_soft"):
        if family == "dense_soft":
            initial = {
                "Theta": theta,
                "A_logits": 0.01
                * jax.random.normal(jax.random.fold_in(key, 51), problem["M"].shape)
                * problem["M"],
            }

            def transition(weights):
                return build_transition_operator(
                    jax.nn.sigmoid(weights["A_logits"]) * problem["M"], cfg
                )["P"]

            def objective(weights, batch):
                return loss_stage2(weights["Theta"], batch, transition(weights), cfg)[0]

            forward = jax.jit(
                lambda weights, batch: forward_stage2(
                    weights["Theta"], batch, transition(weights), cfg
                )["pred"]
            )
        else:
            initial = theta
            p = build_transition_operator(
                jnp.asarray(graphs[family], dtype=jnp.float32), cfg
            )["P"]
            objective = lambda weights, batch: loss_stage2(weights, batch, p, cfg)[0]
            forward = jax.jit(
                lambda weights, batch: forward_stage2(weights, batch, p, cfg)["pred"]
            )
        optimizer, update = make_update(
            objective, cfg["lr_stage1"], cfg["grad_clip_norm"]
        )
        weights, history = fit(
            initial,
            update,
            optimizer,
            draw,
            lambda weights: score(lambda b: forward(weights, b), validation)["mse"],
            steps,
            schedule["eval_every"],
            progress=lambda step, value: progress(
                f"{family} step {step}: validation MSE {value:.4f}"
            ),
        )
        if family == "dense_soft":
            p = transition(weights)
            # Retain the selected adjacency logits as well as the inference artifact.
            np.savez_compressed(
                output / "dense_soft_adjacency_logits.npz",
                A_logits=np.asarray(weights["A_logits"]),
            )
            weights = weights["Theta"]
        entry = save_model(
            output,
            f"{family}-{spec['seed']}",
            weights,
            p,
            spec["model"],
            family=family,
            seed=spec["seed"],
            graph_kind="weighted" if family == "dense_soft" else "binary",
        )
        entries.append(entry)
        selected.append((family, weights, p))
        report["families"][family] = history
    test = batches_for_key(test_key, schedule["test_batches"], problem, eval_cfg)
    for family, weights, p in selected:
        forward = jax.jit(lambda b: forward_stage2(weights, b, p, cfg)["pred"])
        report["families"][family]["test"] = score(forward, test)
    catalog = {
        "schema_version": 1,
        "problem": {"path": "problem.npz", "sha256": sha256(output / "problem.npz")},
        "configuration_origin": "New, explicitly configured baseline training.",
        "models": entries,
    }
    (output / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    (output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
