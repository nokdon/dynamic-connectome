import jax.numpy as jnp
from jax import random


def sample_st(key, reachable_pairs, batch_size):
    """
    reachable_pairs: int32 array of shape [K, 2], each row is (s, t)
    returns:
        s_idx: int32 [B], t_idx: int32 [B]
    """
    K = reachable_pairs.shape[0]
    choice = random.randint(key, shape=(batch_size,), minval=0, maxval=K)
    st = reachable_pairs[choice]  # [B, 2]
    s_idx = st[:, 0]
    t_idx = st[:, 1]
    return s_idx, t_idx


def sample_payload(key, batch_size, D, dtype):
    """
    returns:
        payload: [B, D]
    """
    return random.normal(key, shape=(batch_size, D), dtype=dtype)


def build_node_inputs(q, s_idx, t_idx, payload, cfg):
    """
    q:       [N, 2]
    s_idx:   [B]
    t_idx:   [B]
    payload: [B, D]

    returns:
        node_inputs: [B, N, D+4]
    """
    dtype = cfg["dtype"]
    B = payload.shape[0]
    N = q.shape[0]

    node_ids = jnp.arange(N, dtype=jnp.int32)[None, :]  # [1, N]

    s_mask = (node_ids == s_idx[:, None]).astype(dtype)  # [B, N]
    t_mask = (node_ids == t_idx[:, None]).astype(dtype)  # [B, N]

    payload_block = s_mask[..., None] * payload[:, None, :]  # [B, N, D]
    s_flag = s_mask[..., None]  # [B, N, 1]
    t_flag = t_mask[..., None]  # [B, N, 1]
    q_block = jnp.broadcast_to(q[None, :, :], (B, N, 2))  # [B, N, 2]

    node_inputs = jnp.concatenate(
        [payload_block, s_flag, t_flag, q_block],
        axis=-1,
    )  # [B, N, D+4]

    return node_inputs


def sample_episode_batch(key, q, reachable_pairs, cfg):
    """
    q:               [N, 2]
    reachable_pairs: [K, 2]

    returns dict with:
        node_inputs: [B, N, D+4]
        payload:     [B, D]
        targets:     [B, D]
        s_idx:       [B]
        t_idx:       [B]
    """
    dtype = cfg["dtype"]
    B = cfg["batch_size"]
    D = cfg["D"]

    key_st, key_payload = random.split(key, 2)

    s_idx, t_idx = sample_st(key_st, reachable_pairs, B)  # [B], [B]
    payload = sample_payload(key_payload, B, D, dtype)  # [B, D]
    node_inputs = build_node_inputs(q, s_idx, t_idx, payload, cfg)  # [B, N, D+4]

    batch = {
        "node_inputs": node_inputs,
        "payload": payload,
        "targets": payload,
        "s_idx": s_idx,
        "t_idx": t_idx,
    }
    return batch


__all__ = ["sample_st", "sample_payload", "build_node_inputs", "sample_episode_batch"]
