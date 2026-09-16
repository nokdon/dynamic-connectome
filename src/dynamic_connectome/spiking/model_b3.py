import jax
import jax.numpy as jnp
from jax import lax

from .input_coding import build_b3_node_inputs, make_input_scales
from .spike_surrogate import spike_fn


def build_tgt_repr(U_all, S_all, batch, cfg_b3, params_b3):
    B = batch["t_idx"].shape[0]
    batch_ids = jnp.arange(B, dtype=jnp.int32)
    t_idx = batch["t_idx"]

    readout_mode = cfg_b3.get("readout_mode", "avg_last_k_mem")
    k = int(cfg_b3.get("readout_k", 4))
    U_tail = U_all[-k:]
    k_eff = U_tail.shape[0]

    if readout_mode == "last_mem":
        base_repr = U_all[-1][batch_ids, t_idx, :]
    elif readout_mode == "avg_last_k_mem":
        base_repr = jnp.mean(U_tail, axis=0)[batch_ids, t_idx, :]
    elif readout_mode == "learned_weighted_last_k_mem":
        readout_logits = params_b3["readout_logits"]
        n_logits = readout_logits.shape[0]
        if n_logits < k_eff:
            pad = jnp.zeros((k_eff - n_logits,), dtype=readout_logits.dtype)
            readout_logits = jnp.concatenate([pad, readout_logits], axis=0)
        elif n_logits > k_eff:
            readout_logits = readout_logits[-k_eff:]
        weights = jax.nn.softmax(readout_logits, axis=0)
        U_weighted = jnp.einsum("k,kbnh->bnh", weights, U_tail)
        base_repr = U_weighted[batch_ids, t_idx, :]
    else:
        raise ValueError(f"Unknown readout_mode={readout_mode}")

    if cfg_b3.get("readout_spike_count_feature", False):
        spike_k = min(max(1, int(cfg_b3.get("late_aux_k", k))), S_all.shape[0])
        spike_count = jnp.mean(S_all[-spike_k:], axis=0)[batch_ids, t_idx, :]
        base_repr = base_repr + spike_count @ params_b3["W_spike_readout"]

    if cfg_b3.get("readout_residual_mlp", False):
        hidden = jax.nn.gelu(base_repr @ params_b3["W_ro1"] + params_b3["b_ro1"])
        base_repr = base_repr + (hidden @ params_b3["W_ro2"] + params_b3["b_ro2"])

    return base_repr


def forward_b3_refine(params_b3, batch, P_star, cfg, cfg_b3):
    dtype = cfg["dtype"]
    alpha = jnp.asarray(cfg_b3["alpha_mem"], dtype=dtype)
    T_steps = int(cfg_b3["T_steps"])

    X0 = build_b3_node_inputs(batch["node_inputs"], cfg, cfg_b3).astype(dtype)

    W_enc = params_b3["W_enc"]
    b_enc = params_b3["b_enc"]
    W_msg = params_b3["W_msg"]
    W_msg_u = params_b3["W_msg_u"]
    W_self = params_b3["W_self"]
    b = params_b3["b"]
    theta = params_b3["theta"]
    W_dec = params_b3["W_dec"]
    b_dec = params_b3["b_dec"]

    B, N, _ = X0.shape
    H = W_msg.shape[0]
    J0 = X0 @ W_enc + b_enc
    scales = make_input_scales(T_steps, cfg_b3, dtype)
    P_T = P_star.T.astype(dtype)

    hybrid_channel = bool(cfg_b3.get("hybrid_channel", False))
    hybrid_u_scale_value = cfg_b3.get("hybrid_u_scale", 0.0)
    use_hybrid_u = hybrid_channel and (hybrid_u_scale_value > 0.0)
    hybrid_u_scale = jnp.asarray(hybrid_u_scale_value, dtype=dtype)

    trace_channel = bool(cfg_b3.get("trace_channel", False))
    beta_trace = jnp.asarray(cfg_b3.get("beta_trace", 0.80), dtype=dtype)

    U0 = jnp.zeros((B, N, H), dtype=dtype)
    I0 = jnp.zeros((B, N, H), dtype=dtype)

    def _step(carry, scale_t):
        U, I_syn = carry
        S = spike_fn(U, theta, cfg_b3)

        incoming = jnp.einsum("ij,bjh->bih", P_T, S @ W_msg)

        if use_hybrid_u:
            U_trace = jnp.tanh(U)
            incoming_u = jnp.einsum("ij,bjh->bih", P_T, U_trace @ W_msg_u)
            incoming = incoming + hybrid_u_scale * incoming_u

        if trace_channel:
            incoming_trace = jnp.einsum(
                "ij,bjh->bih", P_T, S @ params_b3["W_msg_trace"]
            )
            I_syn = beta_trace * I_syn + incoming_trace
            incoming = incoming + I_syn

        self_term = U @ W_self
        J_t = scale_t * J0
        U_next = alpha * U * (1.0 - S) + incoming + self_term + b + J_t
        return (U_next, I_syn), (U_next, S)

    (_, I_last), (U_seq, S_seq) = lax.scan(_step, (U0, I0), scales)

    U_all = jnp.concatenate([U0[None, ...], U_seq], axis=0)
    S_all = S_seq
    tgt_repr = build_tgt_repr(U_all, S_all, batch, cfg_b3, params_b3)
    pred = tgt_repr @ W_dec + b_dec

    k_aux = int(cfg_b3.get("late_aux_k", 0))
    late_preds = None
    if k_aux > 0:
        late_pred_list = []
        base_readout_k = int(cfg_b3.get("readout_k", 4))
        for tau in range(k_aux):
            end_u = U_all.shape[0] - (k_aux - 1 - tau)
            end_s = S_all.shape[0] - (k_aux - 1 - tau)
            local_cfg = dict(cfg_b3)
            local_cfg["readout_k"] = min(base_readout_k, tau + 1)
            step_repr = build_tgt_repr(
                U_all[:end_u], S_all[:end_s], batch, local_cfg, params_b3
            )
            late_pred_list.append(step_repr @ W_dec + b_dec)

        late_preds = jnp.stack(late_pred_list, axis=0)

    return {
        "U_all": U_all,
        "S_all": S_all,
        "I_syn_last": I_last,
        "tgt_repr": tgt_repr,
        "pred": pred,
        "late_preds": late_preds,
    }


__all__ = ["build_tgt_repr", "forward_b3_refine"]
