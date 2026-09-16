"""Small, explicit training protocol with post-update validation selection.

This protocol is new. It does not reproduce the historical training-batch
minimum selection used for the supplied checkpoints.
"""

import json
from pathlib import Path
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .artifacts import ARTIFACT_ROOT, load_problem, read_catalog, save_model, sha256
from .config import load_training_config, runtime_config
from .evaluation import batches_for_key, graph_diagnostics, score
from .stage12.init_params import init_params_b1
from .stage12.loss_stage1 import loss_stage1
from .stage12.loss_stage2 import loss_stage2
from .stage12.model_stage2 import forward_stage2, init_params_stage2_from_stage1
from .stage12.router import extract_final_graph
from .stage12.sampler import sample_episode_batch
from .stage12.transition import build_transition_operator
from .spiking.init_b3 import init_params_b3_refine
from .spiking.model_b3 import forward_b3_refine
from .spiking.loss_b3 import loss_b3_next, teacher_coeffs


def make_update(loss_fn, learning_rate, clip, weight_decay=0.0):
    optimizer = optax.chain(
        optax.clip_by_global_norm(clip),
        optax.adamw(learning_rate, weight_decay=weight_decay),
    )

    @jax.jit
    def update(params, state, batch, *extra):
        loss, grad = jax.value_and_grad(loss_fn)(params, batch, *extra)
        updates, state = optimizer.update(grad, state, params)
        return optax.apply_updates(params, updates), state, loss

    return optimizer, update


def fit(
    params,
    update,
    optimizer,
    draw,
    validate,
    steps,
    eval_every,
    *,
    extra=None,
    progress=None,
):
    """Validation sees the actual candidate weights, including the initial state."""
    state = optimizer.init(params)
    best_params = params
    best_value = float(validate(params))
    if not np.isfinite(best_value):
        raise FloatingPointError("Nonfinite initial validation score")
    best_step = 0
    history = [{"step": 0, "validation_mse": best_value}]
    for step in range(1, steps + 1):
        args = () if extra is None else extra(step)
        params, state, training_loss = update(params, state, draw(step), *args)
        if not np.isfinite(float(training_loss)):
            raise FloatingPointError(f"Nonfinite training loss at step {step}")
        if step % eval_every == 0 or step == steps:
            value = float(validate(params))
            if not np.isfinite(value):
                raise FloatingPointError(f"Nonfinite validation loss at step {step}")
            history.append(
                {
                    "step": step,
                    "validation_mse": value,
                    "training_objective_before_update": float(training_loss),
                }
            )
            if value < best_value:
                best_params, best_value, best_step = params, value, step
            if progress is not None:
                progress(step, value)
    return best_params, {
        "selected_step": best_step,
        "validation_mse": best_value,
        "history": history,
    }


def train(config_path, output, *, progress=print):
    spec = load_training_config(config_path)
    cfg = runtime_config(spec["model"])
    sp = {**spec["spiking"], "seed": spec["seed"]}
    spec["spiking"] = sp
    schedule = spec["schedule"]
    problem = load_problem()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(spec, indent=2) + "\n")
    shutil.copyfile(ARTIFACT_ROOT / "problem.npz", output / "problem.npz")
    base = jax.random.PRNGKey(spec["seed"])
    stream_ids = {
        "initialization": 0,
        "stage1_training": 1,
        "stage2_training": 2,
        "spiking_training": 3,
        "validation": 100,
        "test": 200,
    }

    def stream(name):
        return jax.random.fold_in(base, stream_ids[name])

    def draw_for(name):
        key = stream(name)
        return lambda step: sample_episode_batch(
            jax.random.fold_in(key, step), problem["q"], problem["reachable_pairs"], cfg
        )

    eval_cfg = {**cfg, "batch_size": schedule["eval_batch_size"]}
    validation = batches_for_key(
        stream("validation"), schedule["validation_batches"], problem, eval_cfg
    )
    params = init_params_b1(stream("initialization"), problem["M"], cfg)
    # Structural selection follows the original declared reach/density objective;
    # validation payloads are used only for Stage2 and hybrid selection.
    opt, update = make_update(
        lambda p, b: loss_stage1(p, jax.random.PRNGKey(0), b, problem["M"], cfg)[0],
        cfg["lr_stage1"],
        cfg["grad_clip_norm"],
    )
    state = opt.init(params)
    draw = draw_for("stage1_training")
    best_params = None
    best_value = -np.inf
    structural_history = []
    for step in range(1, schedule["stage1_steps"] + 1):
        params, state, loss = update(params, state, draw(step))
        if not np.isfinite(float(loss)):
            raise FloatingPointError(f"Nonfinite Stage1 objective at step {step}")
        if step % schedule["eval_every"] == 0 or step == schedule["stage1_steps"]:
            z = extract_final_graph(params, problem["M"], cfg)["z_star"]
            p = build_transition_operator(z, cfg)["P"]
            diag = graph_diagnostics(p, problem, cfg["L_msg"])
            value = diag["reachable_pair_fraction"] - cfg[
                "score_density_penalty"
            ] * diag["edges"] / float(problem["M"].sum())
            structural_history.append({"step": step, "selection_score": value, **diag})
            if value > best_value:
                best_params, best_value, best_step = params, value, step
            progress(f"Stage1 step {step}: structural score {value:.4f}")
    z = extract_final_graph(best_params, problem["M"], cfg)["z_star"]
    p = build_transition_operator(z, cfg)["P"]
    np.savez_compressed(
        output / "stage1_selected.npz",
        **{k: np.asarray(v) for k, v in best_params.items()},
    )
    report = {
        "protocol": "Stage1 structural selection; Stage2 and hybrid post-update validation MSE; final test evaluated after all selection.",
        "seed": spec["seed"],
        "stream_ids": stream_ids,
        "key_derivation": "fold_in(PRNGKey(seed), stream_id); training: fold_in(stream_key, one_based_step); evaluation: split(stream_key, batch_count)",
        "stage1": {
            "selected_step": best_step,
            "selection_score": best_value,
            "history": structural_history,
        },
    }
    stage2 = init_params_stage2_from_stage1(best_params)
    predict2 = jax.jit(lambda weights, b: forward_stage2(weights, b, p, cfg)["pred"])
    opt2, update2 = make_update(
        lambda weights, b: loss_stage2(weights, b, p, cfg)[0],
        cfg["lr_stage2"],
        cfg["grad_clip_norm"],
    )
    stage2, report["stage2"] = fit(
        stage2,
        update2,
        opt2,
        draw_for("stage2_training"),
        lambda weights: score(lambda b: predict2(weights, b), validation)["mse"],
        schedule["stage2_steps"],
        schedule["eval_every"],
        progress=lambda step, value: progress(
            f"Stage2 step {step}: validation MSE {value:.4f}"
        ),
    )

    # The teacher is a fixed argument of the training loss; inference has no teacher.
    def hybrid_loss(weights, batch, coefficients):
        teacher_out = forward_stage2(stage2, batch, p, cfg)
        teacher = {
            "pred": jax.lax.stop_gradient(teacher_out["pred"]),
            "target_repr": jax.lax.stop_gradient(teacher_out["H_tgt"]),
        }
        return loss_b3_next(weights, batch, p, cfg, sp, teacher, coefficients)[0]

    hybrid = init_params_b3_refine(stage2, cfg, sp)
    opt3, update3 = make_update(
        hybrid_loss, sp["lr_b3"], sp["grad_clip_norm_b3"], sp["weight_decay"]
    )
    predict3 = jax.jit(
        lambda weights, b: forward_b3_refine(weights, b, p, cfg, sp)["pred"]
    )
    bpe = schedule["spiking_batches_per_epoch"]
    hybrid, report["hybrid"] = fit(
        hybrid,
        update3,
        opt3,
        draw_for("spiking_training"),
        lambda weights: score(lambda b: predict3(weights, b), validation)["mse"],
        schedule["spiking_epochs"] * bpe,
        schedule["eval_every"] * bpe,
        extra=lambda step: (
            jnp.asarray(teacher_coeffs((step - 1) // bpe + 1, sp), dtype=jnp.float32),
        ),
        progress=lambda step, value: progress(
            f"Hybrid step {step}: validation MSE {value:.4f}"
        ),
    )
    # No test draw is generated before selection has finished.
    test = batches_for_key(stream("test"), schedule["test_batches"], problem, eval_cfg)
    report["stage2"]["test"] = score(lambda b: predict2(stage2, b), test)
    report["hybrid"]["test"] = score(lambda b: predict3(hybrid, b), test)
    report["zero_test"] = score(lambda b: jnp.zeros_like(b["targets"]), test)
    report["unrestricted_sum_test"] = score(
        lambda b: jnp.sum(b["node_inputs"][:, :, : cfg["D"]], axis=1), test
    )
    report["graph"] = graph_diagnostics(p, problem, cfg["L_msg"])
    catalog = {
        "schema_version": 1,
        "problem": {"path": "problem.npz", "sha256": sha256(output / "problem.npz")},
        "configuration_origin": "Fully specified before this new training run. See config.json and training_report.json.",
        "models": [
            save_model(
                output,
                "stage2",
                stage2,
                p,
                spec["model"],
                family="learned",
                seed=spec["seed"],
            ),
            save_model(
                output,
                "hybrid",
                hybrid,
                p,
                spec["model"],
                family="hybrid",
                seed=spec["seed"],
                spiking_config=sp,
            ),
        ],
    }
    (output / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    (output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
