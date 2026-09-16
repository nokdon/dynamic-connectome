import jax
import jax.numpy as jnp


def apply_activation(x, cfg):
    act = cfg.get("activation", "gelu")

    if act == "relu":
        return jax.nn.relu(x)
    elif act == "tanh":
        return jnp.tanh(x)
    elif act == "gelu":
        return jax.nn.gelu(x)
    else:
        raise ValueError(f"Unknown activation: {act}")


def forward_stage1(params, batch, P, cfg):
    """
    params:
        S_logits   [N, N]
        W_enc      [D+4, H]
        b_enc      [H]
        W_msg      [L_msg, H, H]
        W_self     [L_msg, H, H]
        b_layers   [L_msg, H]
        W_dec      [H, D]
        b_dec      [D]

    batch:
        node_inputs [B, N, D+4]
        t_idx       [B]

    P:
        [N, N]

    returns dict with:
        H0      [B, N, H]
        H_last  [B, N, H]
        H_tgt   [B, H]
        pred    [B, D]
    """
    X0 = batch["node_inputs"]  # [B, N, D+4]
    t_idx = batch["t_idx"]  # [B]

    W_enc = params["W_enc"]  # [D+4, H]
    b_enc = params["b_enc"]  # [H]
    W_msg = params["W_msg"]  # [L_msg, H, H]
    W_self = params["W_self"]  # [L_msg, H, H]
    b_layers = params["b_layers"]  # [L_msg, H]
    W_dec = params["W_dec"]  # [H, D]
    b_dec = params["b_dec"]  # [D]

    B = X0.shape[0]
    L_msg = cfg["L_msg"]

    # ----- encoder -----
    H = X0 @ W_enc + b_enc  # [B, N, H]
    H0 = H

    # incoming aggregation uses P^T because z_ij means i -> j
    P_T = P.T  # [N, N]

    # ----- message-passing stack -----
    for ell in range(L_msg):
        # aggregated incoming hidden states:
        # agg[b, i, :] = sum_j P_{j i} H[b, j, :]
        agg = jnp.einsum("ij,bjh->bih", P_T, H)  # [B, N, H]

        msg_term = agg @ W_msg[ell]  # [B, N, H]
        self_term = H @ W_self[ell]  # [B, N, H]

        H = apply_activation(
            msg_term + self_term + b_layers[ell],
            cfg,
        )  # [B, N, H]

    H_last = H

    # ----- read target node -----
    batch_ids = jnp.arange(B, dtype=jnp.int32)
    H_tgt = H_last[batch_ids, t_idx, :]  # [B, H]

    # ----- decoder -----
    pred = H_tgt @ W_dec + b_dec  # [B, D]

    return {
        "H0": H0,
        "H_last": H_last,
        "H_tgt": H_tgt,
        "pred": pred,
    }


__all__ = ["apply_activation", "forward_stage1"]
