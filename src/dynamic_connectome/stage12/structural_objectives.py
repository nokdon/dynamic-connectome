import jax.numpy as jnp


def build_noisy_or_connectivity_surrogate(p_edge, cfg):
    dtype = p_edge.dtype
    eps = jnp.asarray(cfg["eps"], dtype=dtype)
    L_msg = int(cfg["L_msg"])

    R_prev = jnp.clip(p_edge, 0.0, 1.0)
    R_list = [R_prev]

    for _ in range(1, L_msg):
        x = R_prev[:, :, None] * p_edge[None, :, :]
        x = jnp.clip(x, 0.0, 1.0 - eps)
        R_next = 1.0 - jnp.prod(1.0 - x, axis=1)
        R_next = jnp.clip(R_next, 0.0, 1.0)
        R_next = jnp.maximum(R_next, R_prev)
        R_list.append(R_next)
        R_prev = R_next

    return {"R_list": R_list, "R_final": R_prev}


def build_unnormalized_reach_surrogate(p_edge, cfg):
    dtype = p_edge.dtype
    L_msg = int(cfg["L_msg"])
    beta = jnp.asarray(cfg.get("unnorm_reach_beta", 1.0), dtype=dtype)
    pow_clip = jnp.asarray(cfg.get("pow_clip_value", 20.0), dtype=dtype)

    A = jnp.clip(p_edge, 0.0, 1.0)
    Pk = A
    Xi = A

    for _ in range(1, L_msg):
        Pk = jnp.matmul(Pk, A)
        Pk = jnp.clip(Pk, 0.0, pow_clip)
        Xi = jnp.clip(Xi + Pk, 0.0, pow_clip)

    rho = 1.0 - jnp.exp(-beta * Xi)
    rho = jnp.clip(rho, 0.0, 1.0)

    return {
        "A": A,
        "Xi": Xi,
        "rho": rho,
    }


def gather_conn_st(mat, s_idx, t_idx):
    return mat[s_idx, t_idx]


def compute_router_entropy_penalty(alpha_real_slots, alpha_null_slots, row_valid, cfg):
    dtype = alpha_real_slots.dtype
    eps = jnp.asarray(cfg["eps"], dtype=dtype)

    alpha_aug = jnp.concatenate(
        [alpha_real_slots, alpha_null_slots[:, :, None]], axis=2
    )  # [N,2,N+1]
    slot_entropy = -jnp.sum(alpha_aug * jnp.log(alpha_aug + eps), axis=2)
    denom = jnp.sum(row_valid) * alpha_real_slots.shape[1] + eps
    L_entropy = jnp.sum(slot_entropy * row_valid[:, None]) / denom

    return {
        "L_entropy": L_entropy,
        "slot_entropy": slot_entropy,
    }


def compute_open_extra_penalty(row_open_soft, expected_real_edges, cfg):
    dtype = row_open_soft.dtype
    L_open = jnp.mean(row_open_soft)
    extra_soft = jnp.clip(expected_real_edges - row_open_soft, 0.0, 1.0)
    L_extra = jnp.mean(extra_soft)
    L_row_sparse = L_open + L_extra
    return {
        "L_open": L_open,
        "L_extra": L_extra,
        "L_row_sparse": L_row_sparse,
        "extra_soft": extra_soft,
    }
