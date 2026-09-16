import jax.numpy as jnp
from .model_stage1 import apply_activation
from .router import extract_final_graph
from .transition import build_transition_operator

COMP_KEYS = ("W_enc", "b_enc", "W_msg", "W_self", "b_layers", "W_dec", "b_dec")


def init_params_stage2_from_stage1(params_stage1):
    return {k: params_stage1[k] for k in COMP_KEYS}


def build_transition_from_extracted(params_stage1, mask, cfg):
    ext = extract_final_graph(params_stage1, mask, cfg)
    z_star = ext["z_star"]
    op = build_transition_operator(z_star, cfg)

    return {
        "z_star": z_star,
        "P_star": op["P"],
        "row_open_star": ext["row_open_star"],
        "row_open_soft": ext["row_open_soft"],
        "expected_real_edges": ext["expected_real_edges"],
        "alpha_edge": ext["alpha_edge"],
        "alpha_real_slots": ext["alpha_real_slots"],
        "alpha_null_slots": ext["alpha_null_slots"],
        "top1_prob_slots": ext["top1_prob_slots"],
        "top1_margin_slots": ext["top1_margin_slots"],
    }


def forward_stage2(params_stage2, batch, P_star, cfg):
    X0 = batch["node_inputs"]
    t_idx = batch["t_idx"]

    W_enc = params_stage2["W_enc"]
    b_enc = params_stage2["b_enc"]
    W_msg = params_stage2["W_msg"]
    W_self = params_stage2["W_self"]
    b_layers = params_stage2["b_layers"]
    W_dec = params_stage2["W_dec"]
    b_dec = params_stage2["b_dec"]

    H = X0 @ W_enc + b_enc
    P_T = P_star.T

    for ell in range(cfg["L_msg"]):
        agg = jnp.einsum("ij,bjh->bih", P_T, H)
        msg_term = agg @ W_msg[ell]
        self_term = H @ W_self[ell]
        H = apply_activation(msg_term + self_term + b_layers[ell], cfg)

    batch_ids = jnp.arange(X0.shape[0], dtype=jnp.int32)
    H_tgt = H[batch_ids, t_idx, :]
    pred = H_tgt @ W_dec + b_dec

    return {
        "H_last": H,
        "H_tgt": H_tgt,
        "pred": pred,
    }


__all__ = [
    "COMP_KEYS",
    "init_params_stage2_from_stage1",
    "build_transition_from_extracted",
    "forward_stage2",
]
