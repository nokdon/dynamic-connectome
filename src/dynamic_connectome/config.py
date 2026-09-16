"""Explicit JSON configurations for new, validation-selected experiments."""

import json
import math
from pathlib import Path

import jax.numpy as jnp


def runtime_config(config):
    result = dict(config)
    if result.get("dtype") != "float32":
        raise ValueError("Only explicit float32 configurations are supported")
    result["dtype"] = jnp.float32
    return result


def load_training_config(path):
    config = json.loads(Path(path).read_text())
    model, spike, schedule = (config[k] for k in ("model", "spiking", "schedule"))
    runtime_config(model)
    for section, keys in [
        (model, ("N", "D", "H", "L_msg", "batch_size")),
        (spike, ("T_steps", "readout_k")),
        (
            schedule,
            (
                "stage1_steps",
                "stage2_steps",
                "spiking_epochs",
                "spiking_batches_per_epoch",
                "eval_every",
                "validation_batches",
                "test_batches",
                "eval_batch_size",
            ),
        ),
    ]:
        for key in keys:
            if type(section[key]) is not int or section[key] <= 0:
                raise ValueError(f"{key} must be a positive integer")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be a uint32 integer")
    if model["N"] != 24 or model["k_router"] != 2:
        raise ValueError(
            "This experiment uses the supplied 24-node problem and two-slot router"
        )
    if not 0 <= model["open_budget_topm"] <= model["N"]:
        raise ValueError("Invalid open-row budget")
    if not 1 <= spike["readout_k"] <= spike["T_steps"] + 1:
        raise ValueError("readout_k exceeds the available membrane history")
    if not 0 <= spike["late_aux_k"] <= spike["T_steps"]:
        raise ValueError("late_aux_k exceeds the available spike history")
    if model["activation"] not in ("gelu", "relu", "tanh"):
        raise ValueError("Unsupported activation")
    for section in (model, spike, schedule):
        for key, value in section.items():
            if type(value) is float and not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
    for key in (
        "eps",
        "tau_router",
        "beta_gate",
        "lr_stage1",
        "lr_stage2",
        "grad_clip_norm",
    ):
        if model[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if spike["lr_b3"] <= 0 or spike["grad_clip_norm_b3"] <= 0:
        raise ValueError("Spiking optimizer settings must be positive")
    return config
