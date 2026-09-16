"""Shared synthetic draws, explicit evaluation protocol, no checkpoint selection."""

import platform

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import ARTIFACT_ROOT, load_model, load_problem, read_catalog
from .stage12.sampler import sample_episode_batch
from .stage12.model_stage2 import forward_stage2
from .spiking.model_b3 import forward_b3_refine


def batches_for_key(key, count, problem, cfg):
    return [
        sample_episode_batch(k, problem["q"], problem["reachable_pairs"], cfg)
        for k in jax.random.split(key, count)
    ]


def prediction_function(params, p, cfg, spiking_config=None):
    if spiking_config is None:
        return jax.jit(lambda batch: forward_stage2(params, batch, p, cfg)["pred"])
    return jax.jit(
        lambda batch: forward_b3_refine(params, batch, p, cfg, spiking_config)["pred"]
    )


def score(predict, batches):
    losses = np.concatenate(
        [
            np.asarray(jnp.mean((predict(b) - b["targets"]) ** 2, axis=-1))
            for b in batches
        ]
    )
    if not np.isfinite(losses).all():
        raise FloatingPointError("Nonfinite evaluation loss")
    return {
        "mse": float(np.mean(losses, dtype=np.float64)),
        "sample_loss_std": float(np.std(losses, dtype=np.float64)),
        "examples": int(len(losses)),
    }


def graph_diagnostics(p, problem, hops):
    z = np.asarray(p) > 0
    reach = np.eye(len(z), dtype=bool)
    for _ in range(hops):
        reach |= (reach.astype(np.int32) @ z.astype(np.int32)) > 0
    pairs = np.asarray(problem["reachable_pairs"])
    fraction = float(reach[pairs[:, 0], pairs[:, 1]].mean())
    return {
        "edges": int(z.sum()),
        "open_rows": int(z.any(axis=1).sum()),
        "reachable_pair_fraction": fraction,
        "hops": hops,
    }


def evaluate(
    root=ARTIFACT_ROOT, *, seed=20260916, batch_size=256, batches=16, model_ids=None
):
    if batch_size <= 0 or batches <= 0 or not 0 <= seed < 2**32:
        raise ValueError("Positive batch sizes/counts and a uint32 seed are required")
    catalog = read_catalog(root)
    selected = [m["id"] for m in catalog["models"]] if model_ids is None else model_ids
    if not selected:
        raise ValueError("No models selected")
    problem = load_problem(root, catalog)
    _, _, cfg, _ = load_model(selected[0], root, catalog)
    cfg = {**cfg, "batch_size": batch_size}
    draws = batches_for_key(jax.random.PRNGKey(seed), batches, problem, cfg)
    results = []
    for model_id in selected:
        params, p, local_cfg, entry = load_model(model_id, root, catalog)
        if (local_cfg["D"], local_cfg["N"]) != (cfg["D"], cfg["N"]):
            raise ValueError(
                "Models must share a task to use the same evaluation draws"
            )
        local_cfg = {**local_cfg, "batch_size": batch_size}
        sp = entry["spiking_config"]
        hops = local_cfg["L_msg"] if sp is None else sp["T_steps"]
        result = {
            "id": model_id,
            "family": entry["family"],
            "seed": entry["seed"],
            "artifact_sha256": entry["sha256"],
            **graph_diagnostics(p, problem, hops),
            **score(prediction_function(params, p, local_cfg, sp), draws),
        }
        results.append(result)
    summaries = {}
    for family in sorted({r["family"] for r in results}):
        losses = [r["mse"] for r in results if r["family"] == family]
        summaries[family] = {
            "mean_mse": float(np.mean(losses)),
            "std_across_seeds": float(np.std(losses, ddof=0)),
            "checkpoints": len(losses),
        }
    return {
        "schema_version": 1,
        "protocol": "Evaluation only: fixed checkpoints on common synthetic draws; no tuning.",
        "random_key": seed,
        "batch_size": batch_size,
        "batches": batches,
        "environment": {
            "jax": jax.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "backend": jax.default_backend(),
        },
        "summary": summaries,
        "results": results,
        "zero_baseline": score(lambda b: jnp.zeros_like(b["targets"]), draws),
        "unrestricted_sum": score(
            lambda b: jnp.sum(b["node_inputs"][:, :, : cfg["D"]], axis=1), draws
        ),
    }
