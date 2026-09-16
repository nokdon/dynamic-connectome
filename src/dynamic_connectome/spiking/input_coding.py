import jax.numpy as jnp


def build_b3_node_inputs(node_inputs, cfg, cfg_b3):
    dtype = cfg["dtype"]
    X = node_inputs.astype(dtype)
    D = int(cfg["D"])
    if not cfg_b3.get("signed_payload_code", False):
        return X
    payload = X[..., :D]
    rest = X[..., D:]
    payload_pos = jnp.maximum(payload, 0.0)
    payload_neg = jnp.maximum(-payload, 0.0)
    return jnp.concatenate([payload_pos, payload_neg, rest], axis=-1)


def make_input_scales(T_steps, cfg_b3, dtype):
    mode = cfg_b3.get("input_injection_mode", "constant")
    if mode == "constant":
        return jnp.ones((T_steps,), dtype=dtype)
    if mode == "frontload_decay":
        decay = jnp.asarray(cfg_b3.get("input_decay", 0.75), dtype=dtype)
        idx = jnp.arange(T_steps, dtype=dtype)
        return decay**idx
    if mode == "pulse_then_zero":
        scales = jnp.zeros((T_steps,), dtype=dtype)
        return scales.at[0].set(jnp.asarray(1.0, dtype=dtype))
    raise ValueError(f"Unknown input_injection_mode={mode}")


__all__ = ["build_b3_node_inputs", "make_input_scales"]
