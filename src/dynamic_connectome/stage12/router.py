import jax
import jax.numpy as jnp
from jax import random


def masked_row_softmax(logits, avail_mask, tau, eps=1e-8):
    dtype = logits.dtype
    tau = jnp.asarray(tau, dtype=dtype)
    eps = jnp.asarray(eps, dtype=dtype)
    avail_mask = avail_mask.astype(dtype)

    avail_bool = avail_mask > 0
    row_valid = jnp.any(avail_bool, axis=1).astype(dtype)

    very_neg = jnp.asarray(-1e9, dtype=dtype)
    scaled = logits / jnp.maximum(tau, eps)
    scaled = scaled + jnp.log(jnp.maximum(avail_mask, eps))
    scaled = jnp.where(avail_bool, scaled, very_neg)

    alpha = jax.nn.softmax(scaled, axis=1)
    alpha = alpha * avail_bool.astype(dtype)
    alpha = alpha * row_valid[:, None]
    return alpha, row_valid


def hard_top1_choice(scores, avail_mask):
    dtype = scores.dtype
    avail_mask = avail_mask.astype(dtype)
    avail_bool = avail_mask > 0
    row_valid = jnp.any(avail_bool, axis=1).astype(dtype)
    very_neg = jnp.asarray(-1e9, dtype=dtype)
    masked_scores = jnp.where(avail_bool, scores, very_neg)
    choice_idx = jnp.argmax(masked_scores, axis=1)
    z = jax.nn.one_hot(choice_idx, scores.shape[1], dtype=dtype)
    z = z * row_valid[:, None] * avail_mask
    z = jnp.clip(z, 0.0, 1.0)
    return z, choice_idx, row_valid


def gate_sigmoid_ste(logits, beta, dtype):
    beta = jnp.asarray(beta, dtype=dtype)
    logits = logits.astype(dtype)
    soft = jax.nn.sigmoid(beta * logits)
    hard = (logits >= 0.0).astype(dtype)
    ste = soft + jax.lax.stop_gradient(hard - soft)
    return soft, hard, ste


def hard_topm_rows(scores, row_valid, k_open):
    """
    Deterministic exact-open-row budget.
    Returns a binary mask with exactly min(k_open, n_valid_rows) open rows.
    """
    dtype = scores.dtype
    N = scores.shape[0]
    row_valid = row_valid.astype(dtype)

    if k_open is None:
        return row_valid

    k_open = int(k_open)
    if k_open <= 0:
        return jnp.zeros((N,), dtype=dtype)

    k_eff = min(k_open, N)
    very_neg = jnp.asarray(-1e9, dtype=dtype)
    masked_scores = jnp.where(row_valid > 0, scores, very_neg)

    _, idx = jax.lax.top_k(masked_scores, k_eff)
    topm = jnp.sum(jax.nn.one_hot(idx, N, dtype=dtype), axis=0)
    topm = jnp.clip(topm, 0.0, 1.0)
    topm = topm * row_valid
    return topm


def sequential_hard_rowgate_router(key, params, mask, cfg):
    """
    Row i first decides whether it opens at all.
    Optional cfg["open_budget_topm"] enforces an exact hard number of open rows.
    If open, the row chooses a first real edge by deterministic top-1.
    Then it decides whether to emit a second edge, and if yes chooses a second
    distinct real edge by deterministic top-1 on the residual mask.

    Soft backward still flows through the original sigmoid gates.
    """
    del key
    dtype = cfg["dtype"]
    eps = jnp.asarray(cfg["eps"], dtype=dtype)
    tau_router = jnp.asarray(cfg["tau_router"], dtype=dtype)
    beta_gate = jnp.asarray(cfg.get("beta_gate", 8.0), dtype=dtype)

    S_logits = params["S_logits"]
    open_logits = params["open_logits"]
    extra_logits = params["extra_logits"]
    mask = mask.astype(dtype)

    row_valid = jnp.any(mask > 0, axis=1).astype(dtype)

    # Base gates
    open_soft_base, open_hard_thresh, _ = gate_sigmoid_ste(
        open_logits, beta_gate, dtype
    )
    extra_soft_base, extra_hard_base, _ = gate_sigmoid_ste(
        extra_logits, beta_gate, dtype
    )

    open_soft_base = open_soft_base * row_valid
    open_hard_thresh = open_hard_thresh * row_valid
    extra_soft_base = extra_soft_base * open_soft_base
    extra_hard_base = extra_hard_base * open_hard_thresh

    # Optional exact hard budget on open rows
    k_open = cfg.get("open_budget_topm", None)
    if k_open is None:
        open_hard = open_hard_thresh
    else:
        open_hard = hard_topm_rows(open_logits, row_valid, k_open)

    open_ste = open_soft_base + jax.lax.stop_gradient(open_hard - open_soft_base)

    # Slot 1
    alpha1, _ = masked_row_softmax(S_logits, mask, tau_router, eps=eps)
    z1_hard, idx1, _ = hard_top1_choice(S_logits, mask)

    # Slot 2 with soft/hard without-replacement
    soft_avail2 = mask * (1.0 - alpha1)
    alpha2, _ = masked_row_softmax(S_logits, soft_avail2, tau_router, eps=eps)
    hard_avail2 = mask * (1.0 - z1_hard)
    z2_hard, idx2, _ = hard_top1_choice(S_logits, hard_avail2)

    extra_soft = extra_soft_base * open_soft_base
    extra_hard = extra_hard_base * open_hard
    extra_ste = extra_soft + jax.lax.stop_gradient(extra_hard - extra_soft)

    z1_soft = open_soft_base[:, None] * alpha1
    z2_soft = extra_soft[:, None] * alpha2
    z_soft = jnp.clip(z1_soft + z2_soft, 0.0, 1.0) * mask

    z1_hard = open_hard[:, None] * z1_hard
    z2_hard = extra_hard[:, None] * z2_hard
    z_hard = jnp.clip(z1_hard + z2_hard, 0.0, 1.0) * mask

    z_ste = z_soft + jax.lax.stop_gradient(z_hard - z_soft)

    alpha_real_slots = jnp.stack([z1_soft, z2_soft], axis=1)
    alpha_null_slots = jnp.stack([1.0 - open_soft_base, 1.0 - extra_soft], axis=1)

    expected_real_edges = open_soft_base + extra_soft
    row_open_soft = open_soft_base
    row_open_hard = open_hard

    def _slot_top_stats(alpha_slot, null_mass):
        alpha_aug = jnp.concatenate([alpha_slot, null_mass[:, None]], axis=1)
        top2 = jax.lax.top_k(alpha_aug, min(2, alpha_aug.shape[1]))[0]
        top1 = top2[:, 0]
        margin = jnp.where(top2.shape[1] >= 2, top2[:, 0] - top2[:, 1], top2[:, 0])
        return top1, margin

    top1_1, margin_1 = _slot_top_stats(z1_soft, 1.0 - open_soft_base)
    top1_2, margin_2 = _slot_top_stats(z2_soft, 1.0 - extra_soft)
    top1_prob_slots = jnp.stack([top1_1, top1_2], axis=1)
    top1_margin_slots = jnp.stack([margin_1, margin_2], axis=1)

    return {
        "alpha_real_slots": alpha_real_slots,
        "alpha_null_slots": alpha_null_slots,
        "alpha_edge": z_soft,
        "p_edge": z_soft,
        "z_hard": z_hard,
        "z_ste": z_ste,
        "row_open_soft": row_open_soft,
        "row_open_hard": row_open_hard,
        "expected_real_edges": expected_real_edges,
        "top1_prob_slots": top1_prob_slots,
        "top1_margin_slots": top1_margin_slots,
        "row_valid": row_valid,
        "open_soft": open_soft_base,
        "open_hard": open_hard,
        "extra_soft": extra_soft,
        "extra_hard": extra_hard,
        "idx1": idx1,
        "idx2": idx2,
    }


def sample_binary_graph_ste(key, params, mask, cfg):
    return sequential_hard_rowgate_router(key, params, mask, cfg)


def extract_final_graph(params, mask, cfg):
    out = sequential_hard_rowgate_router(random.PRNGKey(0), params, mask, cfg)
    return {
        "alpha_edge": out["alpha_edge"],
        "alpha_real_slots": out["alpha_real_slots"],
        "alpha_null_slots": out["alpha_null_slots"],
        "z_star": out["z_hard"],
        "row_open_star": out["row_open_hard"],
        "row_open_soft": out["row_open_soft"],
        "expected_real_edges": out["expected_real_edges"],
        "top1_prob_slots": out["top1_prob_slots"],
        "top1_margin_slots": out["top1_margin_slots"],
        "open_soft": out["open_soft"],
        "extra_soft": out["extra_soft"],
    }


__all__ = [
    "masked_row_softmax",
    "hard_top1_choice",
    "gate_sigmoid_ste",
    "hard_topm_rows",
    "sequential_hard_rowgate_router",
    "sample_binary_graph_ste",
    "extract_final_graph",
]
