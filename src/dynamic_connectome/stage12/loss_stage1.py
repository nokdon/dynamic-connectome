import jax.numpy as jnp
from .model_stage1 import forward_stage1
from .router import extract_final_graph, sample_binary_graph_ste
from .structural_objectives import (
    build_noisy_or_connectivity_surrogate,
    build_unnormalized_reach_surrogate,
    compute_open_extra_penalty,
    compute_router_entropy_penalty,
    gather_conn_st,
)
from .transition import build_transition_operator


def loss_stage1(params, key, batch, mask, cfg):
    dtype = cfg["dtype"]
    eps = jnp.asarray(cfg["eps"], dtype=dtype)
    mask = mask.astype(dtype)

    graph_out = sample_binary_graph_ste(key, params, mask, cfg)
    alpha_edge = graph_out["alpha_edge"]
    p_edge = graph_out["p_edge"]
    z_hard = graph_out["z_hard"]
    z_ste = graph_out["z_ste"]
    row_open_soft = graph_out["row_open_soft"]
    row_open_hard = graph_out["row_open_hard"]
    expected_real_edges = graph_out["expected_real_edges"]
    row_valid = graph_out["row_valid"]

    op_out = build_transition_operator(z_ste, cfg)
    P = op_out["P"]
    out_degree_hard = op_out["out_degree"]

    fwd_out = forward_stage1(params, batch, P, cfg)
    pred = fwd_out["pred"]
    targets = batch["targets"]

    per_example_mse = jnp.mean((pred - targets) ** 2, axis=-1)
    L_task = jnp.mean(per_example_mse)

    noisy_out = build_noisy_or_connectivity_surrogate(p_edge, cfg)
    conn_noisy = gather_conn_st(noisy_out["R_final"], batch["s_idx"], batch["t_idx"])
    L_conn_noisy = -jnp.mean(jnp.log(eps + conn_noisy))

    unreach_out = build_unnormalized_reach_surrogate(p_edge, cfg)
    conn_unreach = gather_conn_st(unreach_out["rho"], batch["s_idx"], batch["t_idx"])
    L_conn_unreach = -jnp.mean(jnp.log(eps + conn_unreach))

    objective_mode = cfg.get("objective_mode", "noisy_or")
    if objective_mode == "noisy_or":
        L_struct = cfg.get("lambda_conn_noisy", 1.0) * L_conn_noisy
    elif objective_mode == "unnorm_reach":
        L_struct = cfg.get("lambda_conn_unreach", 1.0) * L_conn_unreach
    elif objective_mode == "hybrid":
        L_struct = (
            cfg.get("lambda_conn_noisy", 1.0) * L_conn_noisy
            + cfg.get("lambda_conn_unreach", 1.0) * L_conn_unreach
        )
    else:
        raise ValueError(f"Unknown objective_mode: {objective_mode}")

    ent_out = compute_router_entropy_penalty(
        graph_out["alpha_real_slots"],
        graph_out["alpha_null_slots"],
        row_valid,
        cfg,
    )
    L_entropy = ent_out["L_entropy"]

    sparse_out = compute_open_extra_penalty(row_open_soft, expected_real_edges, cfg)
    L_open = sparse_out["L_open"]
    L_extra = sparse_out["L_extra"]
    L_row_sparse = sparse_out["L_row_sparse"]

    total_loss = (
        cfg["lambda_task"] * L_task
        + L_struct
        + cfg["lambda_entropy"] * L_entropy
        + cfg["lambda_open"] * L_open
        + cfg["lambda_extra"] * L_extra
    )

    extract_out = extract_final_graph(params, mask, cfg)
    z_star = extract_out["z_star"]

    valid_edge_count = jnp.sum(mask > 0).astype(dtype)
    mean_p_edge = jnp.sum(p_edge) / (valid_edge_count + eps)
    mean_z_hard = jnp.sum(z_hard) / (valid_edge_count + eps)
    mean_z_star = jnp.sum(z_star) / (valid_edge_count + eps)

    mean_expected_edges = jnp.mean(expected_real_edges)
    mean_row_open_soft = jnp.mean(row_open_soft)
    mean_row_open_hard = jnp.mean(row_open_hard)
    mean_null_prob_slot0 = jnp.mean(graph_out["alpha_null_slots"][:, 0])
    mean_null_prob_last = jnp.mean(graph_out["alpha_null_slots"][:, -1])

    top1_prob_mean = jnp.mean(graph_out["top1_prob_slots"])
    top1_margin_mean = jnp.mean(graph_out["top1_margin_slots"])

    in_degree_hard = jnp.sum(z_hard, axis=0)
    src_out_degree_hard = out_degree_hard[batch["s_idx"]]
    tgt_in_degree_hard = in_degree_hard[batch["t_idx"]]

    aux = {
        "total_loss": total_loss,
        "L_task": L_task,
        "L_conn": L_struct,  # chosen structural loss for legacy logs
        "L_conn_noisy": L_conn_noisy,
        "L_conn_unreach": L_conn_unreach,
        "L_entropy": L_entropy,
        "L_open": L_open,
        "L_extra": L_extra,
        "L_row_sparse": L_row_sparse,
        "mean_p_edge": mean_p_edge,
        "mean_z_hard": mean_z_hard,
        "mean_z_star": mean_z_star,
        "mean_expected_edges": mean_expected_edges,
        "mean_row_open_soft": mean_row_open_soft,
        "mean_row_open_hard": mean_row_open_hard,
        "mean_null_prob_slot0": mean_null_prob_slot0,
        "mean_null_prob_last": mean_null_prob_last,
        "conn_st_mean": jnp.mean(conn_noisy),
        "conn_st_min": jnp.min(conn_noisy),
        "conn_unreach_mean": jnp.mean(conn_unreach),
        "conn_unreach_min": jnp.min(conn_unreach),
        "top1_prob_mean": top1_prob_mean,
        "top1_margin_mean": top1_margin_mean,
        "src_out_degree_hard_mean": jnp.mean(src_out_degree_hard),
        "tgt_in_degree_hard_mean": jnp.mean(tgt_in_degree_hard),
        "row_valid_frac": jnp.mean(row_valid),
        "pred": pred,
        "alpha_edge": alpha_edge,
        "p_edge": p_edge,
        "z_hard": z_hard,
        "z_ste": z_ste,
        "z_star": z_star,
        "row_open_soft": row_open_soft,
        "row_open_hard": row_open_hard,
        "expected_real_edges": expected_real_edges,
        "objective_mode": objective_mode,
    }

    return total_loss, aux


__all__ = ["loss_stage1"]
