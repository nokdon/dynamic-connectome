import math

import jax
import jax.numpy as jnp
from jax import random


def init_params_b3_refine(params_stage2, cfg, cfg_b3):
    dtype = cfg["dtype"]
    D = int(cfg["D"])

    W_msg_src = params_stage2["W_msg"]
    W_self_src = params_stage2["W_self"]
    W_msg0 = W_msg_src[0] if getattr(W_msg_src, "ndim", None) == 3 else W_msg_src
    W_self0 = W_self_src[0] if getattr(W_self_src, "ndim", None) == 3 else W_self_src

    W_enc_old = params_stage2["W_enc"].astype(dtype)
    b_enc = params_stage2["b_enc"].astype(dtype)
    W_dec_old = params_stage2["W_dec"].astype(dtype)
    b_dec = params_stage2["b_dec"].astype(dtype)

    if cfg_b3.get("signed_payload_code", False):
        W_payload = W_enc_old[:D, :]
        W_rest = W_enc_old[D:, :]
        W_enc = jnp.concatenate([W_payload, -W_payload, W_rest], axis=0)
    else:
        W_enc = W_enc_old

    theta_init = jnp.asarray(cfg_b3["theta_init"], dtype=dtype)
    H = W_msg0.shape[0]
    theta = (
        jnp.ones((H,), dtype=dtype) * theta_init
        if cfg_b3.get("use_theta_per_channel", True)
        else theta_init
    )

    hybrid_init_scale = jnp.asarray(cfg_b3.get("hybrid_init_scale", 0.10), dtype=dtype)
    params_b3 = {
        "W_enc": W_enc,
        "b_enc": b_enc,
        "W_msg": W_msg0.astype(dtype),
        "W_msg_u": (hybrid_init_scale * W_msg0).astype(dtype),
        "W_self": W_self0.astype(dtype),
        "b": jnp.zeros((H,), dtype=dtype),
        "theta": theta,
        "W_dec": W_dec_old.astype(dtype),
        "b_dec": b_dec,
    }

    if cfg_b3.get("trace_channel", False):
        params_b3["W_msg_trace"] = (0.10 * W_msg0).astype(dtype)

    readout_mode = cfg_b3.get("readout_mode", "avg_last_k_mem")
    if readout_mode == "learned_weighted_last_k_mem":
        k = int(cfg_b3.get("readout_k", 4))
        params_b3["readout_logits"] = jnp.zeros((k,), dtype=dtype)

    if cfg_b3.get("readout_spike_count_feature", False):
        params_b3["W_spike_readout"] = 0.10 * jnp.eye(H, dtype=dtype)

    if cfg_b3.get("readout_residual_mlp", False):
        hidden_mult = float(cfg_b3.get("readout_residual_hidden_mult", 1.5))
        hidden_dim = max(H, int(math.ceil(hidden_mult * H)))
        init_scale = jnp.asarray(
            cfg_b3.get("readout_residual_init_scale", 0.10), dtype=dtype
        )
        key_local = random.PRNGKey(int(cfg_b3.get("seed", 0)) + 12345)
        k1, k2 = random.split(key_local)
        params_b3["W_ro1"] = (
            init_scale * random.normal(k1, (H, hidden_dim), dtype=dtype) / jnp.sqrt(H)
        ).astype(dtype)
        params_b3["b_ro1"] = jnp.zeros((hidden_dim,), dtype=dtype)
        params_b3["W_ro2"] = (
            init_scale
            * random.normal(k2, (hidden_dim, H), dtype=dtype)
            / jnp.sqrt(hidden_dim)
        ).astype(dtype)
        params_b3["b_ro2"] = jnp.zeros((H,), dtype=dtype)

    return params_b3
