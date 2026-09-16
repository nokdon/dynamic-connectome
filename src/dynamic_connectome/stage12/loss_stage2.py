import jax
import jax.numpy as jnp
from .model_stage2 import forward_stage2


def l2_sq_tree(params):
    leaves = jax.tree_util.tree_leaves(params)
    return sum([jnp.sum(x * x) for x in leaves])


def loss_stage2(params_stage2, batch, P_star, cfg):
    fwd = forward_stage2(params_stage2, batch, P_star, cfg)
    pred = fwd["pred"]
    targets = batch["targets"]

    per_example_mse = jnp.mean((pred - targets) ** 2, axis=-1)
    L_task = jnp.mean(per_example_mse)
    L_wd = cfg["weight_decay_stage2"] * l2_sq_tree(params_stage2)

    total_loss = L_task + L_wd

    return total_loss, {
        "L_task": L_task,
        "L_wd": L_wd,
        "pred": pred,
    }


__all__ = ["l2_sq_tree", "loss_stage2"]
