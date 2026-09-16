import jax.numpy as jnp
from jax import random


def glorot_uniform(key, shape, dtype=jnp.float32):
    fan_in = shape[-2]
    fan_out = shape[-1]
    limit = jnp.sqrt(jnp.asarray(6.0 / (fan_in + fan_out), dtype=dtype))
    return random.uniform(
        key,
        shape=shape,
        minval=-limit,
        maxval=limit,
        dtype=dtype,
    )


def init_linear(key, in_dim, out_dim, dtype=jnp.float32, use_bias=True):
    W = glorot_uniform(key, (in_dim, out_dim), dtype=dtype)
    if use_bias:
        b = jnp.zeros((out_dim,), dtype=dtype)
        return W, b
    return W


def init_params_b1(key, mask, cfg):
    dtype = cfg["dtype"]
    D = cfg["D"]
    H = cfg["H"]
    L_msg = cfg["L_msg"]

    support_logit_noise = jnp.asarray(cfg.get("support_logit_noise", 0.25), dtype=dtype)
    open_logit_init_mean = jnp.asarray(
        cfg.get("open_logit_init_mean", 0.0), dtype=dtype
    )
    open_logit_noise = jnp.asarray(cfg.get("open_logit_noise", 0.15), dtype=dtype)
    extra_logit_init_mean = jnp.asarray(
        cfg.get("extra_logit_init_mean", 0.35), dtype=dtype
    )
    extra_logit_noise = jnp.asarray(cfg.get("extra_logit_noise", 0.15), dtype=dtype)

    n_needed = 4 + 2 * L_msg + 2
    keys = random.split(key, n_needed)

    k_struct = keys[0]
    k_open = keys[1]
    k_extra = keys[2]
    k_enc = keys[3]
    k_msg = keys[4 : 4 + L_msg]
    k_self = keys[4 + L_msg : 4 + 2 * L_msg]
    k_dec = keys[-1]

    mask = mask.astype(dtype)

    # Real-edge ranking logits over valid candidates.
    struct_noise = support_logit_noise * random.normal(
        k_struct, mask.shape, dtype=dtype
    )
    S_logits = jnp.where(mask > 0, struct_noise, jnp.asarray(-10.0, dtype=dtype))

    N = mask.shape[0]
    open_logits = open_logit_init_mean + open_logit_noise * random.normal(
        k_open, (N,), dtype=dtype
    )
    extra_logits = extra_logit_init_mean + extra_logit_noise * random.normal(
        k_extra, (N,), dtype=dtype
    )

    W_enc, b_enc = init_linear(k_enc, D + 4, H, dtype=dtype, use_bias=True)

    W_msg_list = []
    W_self_list = []
    b_list = []
    for ell in range(L_msg):
        Wm, _ = init_linear(k_msg[ell], H, H, dtype=dtype, use_bias=True)
        Ws, _ = init_linear(k_self[ell], H, H, dtype=dtype, use_bias=True)
        b = jnp.zeros((H,), dtype=dtype)
        W_msg_list.append(Wm)
        W_self_list.append(Ws)
        b_list.append(b)

    W_msg = jnp.stack(W_msg_list, axis=0)
    W_self = jnp.stack(W_self_list, axis=0)
    b_layers = jnp.stack(b_list, axis=0)

    W_dec, b_dec = init_linear(k_dec, H, D, dtype=dtype, use_bias=True)

    return {
        "S_logits": S_logits,
        "open_logits": open_logits,
        "extra_logits": extra_logits,
        "W_enc": W_enc,
        "b_enc": b_enc,
        "W_msg": W_msg,
        "W_self": W_self,
        "b_layers": b_layers,
        "W_dec": W_dec,
        "b_dec": b_dec,
    }


__all__ = ["glorot_uniform", "init_linear", "init_params_b1"]
