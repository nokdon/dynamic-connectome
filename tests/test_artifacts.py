import json
import shutil

import jax
import numpy as np
import pytest

from dynamic_connectome.artifacts import (
    ARTIFACT_ROOT,
    checked_path,
    load_model,
    load_problem,
    read_catalog,
    save_model,
    sha256,
    validate_model,
)
from dynamic_connectome.evaluation import prediction_function
from dynamic_connectome.stage12.sampler import sample_episode_batch


def test_every_bundled_artifact_passes_semantic_validation():
    catalog = read_catalog()
    assert len(catalog["models"]) == 26
    for entry in catalog["models"]:
        load_model(entry["id"])


def test_round_trip_preserves_predictions(tmp_path):
    params, p, cfg, entry = load_model("hybrid-0")
    shutil.copyfile(ARTIFACT_ROOT / "problem.npz", tmp_path / "problem.npz")
    saved = save_model(
        tmp_path,
        "hybrid",
        params,
        p,
        entry["config"],
        family="hybrid",
        seed=0,
        spiking_config=entry["spiking_config"],
    )
    catalog = {
        "schema_version": 1,
        "problem": {"path": "problem.npz", "sha256": sha256(tmp_path / "problem.npz")},
        "models": [saved],
    }
    (tmp_path / "catalog.json").write_text(json.dumps(catalog))
    actual, p2, cfg2, _ = load_model("hybrid", tmp_path)
    problem = load_problem()
    batch = sample_episode_batch(
        jax.random.PRNGKey(1),
        problem["q"],
        problem["reachable_pairs"],
        {**cfg, "batch_size": 2},
    )
    np.testing.assert_array_equal(
        prediction_function(params, p, cfg, entry["spiking_config"])(batch),
        prediction_function(actual, p2, cfg2, entry["spiking_config"])(batch),
    )
    with pytest.raises(FileExistsError):
        save_model(
            tmp_path,
            "hybrid",
            params,
            p,
            entry["config"],
            family="hybrid",
            seed=0,
            spiking_config=entry["spiking_config"],
        )


def test_corruption_and_catalog_traversal_are_rejected(tmp_path):
    file = tmp_path / "bad.npz"
    file.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        checked_path(tmp_path, {"path": "bad.npz", "sha256": "0" * 64})
    with pytest.raises(ValueError, match="escapes"):
        checked_path(tmp_path, {"path": "../outside.npz", "sha256": "0" * 64})


def test_semantics_reject_wrong_direction_support_and_shapes():
    params, p, _, entry = load_model("learned-0")
    params = {k: np.asarray(v) for k, v in params.items()}
    p = np.asarray(p).copy()
    p[0, 0] = 1
    with pytest.raises(ValueError, match="support"):
        validate_model(params, p, entry, load_problem())
    _, p, _, _ = load_model("learned-0")
    params["W_dec"] = params["W_dec"][:1]
    with pytest.raises(ValueError, match="shapes"):
        validate_model(params, np.asarray(p), entry, load_problem())
