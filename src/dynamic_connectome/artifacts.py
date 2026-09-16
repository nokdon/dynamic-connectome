"""Numeric-only artifacts: explicit shapes, graph semantics and SHA-256 checks."""

import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .config import runtime_config

ARTIFACT_ROOT = Path(__file__).parent / "_artifacts"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_path(root, entry):
    root = Path(root).resolve()
    path = (root / entry["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Artifact path escapes its catalog directory")
    if sha256(path) != entry["sha256"]:
        raise ValueError(f"Artifact checksum mismatch: {path.name}")
    return path


def read_catalog(root=ARTIFACT_ROOT):
    catalog = json.loads((Path(root) / "catalog.json").read_text())
    if catalog["schema_version"] != 1:
        raise ValueError("Unsupported artifact schema")
    ids = [m["id"] for m in catalog["models"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate model IDs")
    return catalog


def load_problem(root=ARTIFACT_ROOT, catalog=None):
    catalog = read_catalog(root) if catalog is None else catalog
    with np.load(checked_path(root, catalog["problem"]), allow_pickle=False) as data:
        q, mask, pairs = (data[k] for k in ("q", "M", "reachable_pairs"))
    n = len(q)
    if (
        q.shape != (n, 2)
        or mask.shape != (n, n)
        or pairs.ndim != 2
        or pairs.shape[1] != 2
    ):
        raise ValueError("Invalid problem shapes")
    if (
        not np.isfinite(q).all()
        or not np.isin(mask, [0, 1]).all()
        or np.diag(mask).any()
    ):
        raise ValueError("Invalid coordinates or candidate graph")
    if (
        not np.issubdtype(pairs.dtype, np.integer)
        or not len(pairs)
        or np.any(pairs < 0)
        or np.any(pairs >= n)
    ):
        raise ValueError("Invalid source-target pairs")
    if np.any(pairs[:, 0] == pairs[:, 1]) or len(np.unique(pairs, axis=0)) != len(
        pairs
    ):
        raise ValueError("Pairs must be distinct, non-self pairs")
    return {
        "q": jnp.asarray(q),
        "M": jnp.asarray(mask),
        "reachable_pairs": jnp.asarray(pairs),
    }


def validate_model(params, p, entry, problem):
    cfg = entry["config"]
    n, d, h, l = (cfg[k] for k in ("N", "D", "H", "L_msg"))
    if n != len(problem["q"]) or p.shape != (n, n):
        raise ValueError("Model/problem shape mismatch")
    if (
        not np.isfinite(p).all()
        or np.any(p < 0)
        or np.any((p > 0) & (np.asarray(problem["M"]) == 0))
    ):
        raise ValueError(
            "Transition violates candidate support or finite nonnegative weights"
        )
    active = p.sum(axis=1) > 0
    if not np.allclose(p.sum(axis=1), active.astype(float), atol=1e-6):
        raise ValueError("Transition rows must sum to one or zero")
    if entry["graph_kind"] not in ("binary", "weighted"):
        raise ValueError("Unknown graph kind")
    if entry["graph_kind"] == "binary":
        z = (p > 0).astype(float)
        expected = z / np.maximum(z.sum(axis=1, keepdims=True), 1)
        if not np.allclose(p, expected, atol=1e-6):
            raise ValueError(
                "Binary graph transition must have uniform outgoing weights"
            )
    if (
        entry["family"] in ("learned", "random", "heuristic", "hybrid")
        and (p > 0).sum(axis=1).max() > 2
    ):
        raise ValueError("Sparse graph exceeds the two-edge row budget")
    if any(not np.isfinite(v).all() or v.dtype != np.float32 for v in params.values()):
        raise ValueError("Parameters must be finite float32 arrays")
    if {k: list(v.shape) for k, v in params.items()} != entry["parameter_shapes"]:
        raise ValueError("Parameter shapes do not match the manifest")
    required = {"b_enc": (h,), "W_dec": (h, d), "b_dec": (d,)}
    if entry["kind"] == "gnn":
        required.update(
            W_enc=(d + 4, h), W_msg=(l, h, h), W_self=(l, h, h), b_layers=(l, h)
        )
    elif entry["kind"] == "hybrid":
        sp = entry["spiking_config"]
        required.update(
            W_enc=((2 * d if sp["signed_payload_code"] else d) + 4, h),
            W_msg=(h, h),
            W_msg_u=(h, h),
            W_self=(h, h),
            b=(h,),
            theta=(h,) if sp["use_theta_per_channel"] else (),
        )
        if sp["trace_channel"]:
            required["W_msg_trace"] = (h, h)
        if sp["readout_mode"] == "learned_weighted_last_k_mem":
            required["readout_logits"] = (sp["readout_k"],)
        if sp["readout_spike_count_feature"]:
            required["W_spike_readout"] = (h, h)
        if sp["readout_residual_mlp"]:
            hh = max(h, int(np.ceil(sp["readout_residual_hidden_mult"] * h)))
            required.update(W_ro1=(h, hh), b_ro1=(hh,), W_ro2=(hh, h), b_ro2=(h,))
    else:
        raise ValueError("Unknown model kind")
    if set(required) != set(params) or any(
        params[k].shape != v for k, v in required.items()
    ):
        raise ValueError("Parameters do not implement the declared model configuration")


def load_model(model_id, root=ARTIFACT_ROOT, catalog=None):
    catalog = read_catalog(root) if catalog is None else catalog
    entries = [m for m in catalog["models"] if m["id"] == model_id]
    if not entries:
        raise ValueError(f"Unknown model {model_id!r}")
    entry = entries[0]
    with np.load(checked_path(root, entry), allow_pickle=False) as data:
        if set(data.files) != {"P"} | {
            "param__" + k for k in entry["parameter_shapes"]
        }:
            raise ValueError("Unexpected model arrays")
        p = data["P"]
        params = {k: data["param__" + k] for k in entry["parameter_shapes"]}
    validate_model(params, p, entry, load_problem(root, catalog))
    return (
        {k: jnp.asarray(v) for k, v in params.items()},
        jnp.asarray(p),
        runtime_config(entry["config"]),
        entry,
    )


def save_model(
    root,
    model_id,
    params,
    p,
    config,
    *,
    family,
    seed,
    spiking_config=None,
    graph_kind="binary",
):
    root = Path(root)
    path = root / "models" / f"{model_id}.npz"
    if not model_id or any(
        ch not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for ch in model_id
    ):
        raise ValueError("Invalid model ID")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement of an experiment.
    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            P=np.asarray(p),
            **{"param__" + k: np.asarray(v) for k, v in params.items()},
        )
    return {
        "id": model_id,
        "family": family,
        "seed": seed,
        "kind": "hybrid" if spiking_config else "gnn",
        "graph_kind": graph_kind,
        "path": str(path.relative_to(root)),
        "sha256": sha256(path),
        "config": config,
        "spiking_config": spiking_config,
        "parameter_shapes": {k: list(v.shape) for k, v in params.items()},
    }
