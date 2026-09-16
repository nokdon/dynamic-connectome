import jax.numpy as jnp


def build_transition_operator(z_graph, cfg):
    """
    z_graph: [N, N]
        z_ste during Stage 1, and z_star during extraction / Stage 2.

    returns dict with:
        out_degree: [N]
        row_active: [N]   (1 if node has at least one outgoing edge)
        P:          [N, N]
    """
    dtype = z_graph.dtype
    eps = jnp.asarray(cfg["eps"], dtype=dtype)

    out_degree = jnp.sum(z_graph, axis=1)
    row_active = (out_degree > 0).astype(dtype)

    P = (z_graph / (out_degree[:, None] + eps)) * row_active[:, None]

    return {
        "out_degree": out_degree,
        "row_active": row_active,
        "P": P,
    }


__all__ = ["build_transition_operator"]
