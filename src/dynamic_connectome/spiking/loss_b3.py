import jax.numpy as jnp

from .model_b3 import forward_b3_refine


def base_distill_warmup(epoch, cfg_b3):
    warm = int(cfg_b3.get("distill_warmup_epochs", 0))
    if warm <= 0:
        return 1.0
    return min(float(epoch) / float(warm), 1.0)


def teacher_coeffs(epoch, cfg_b3):
    base = base_distill_warmup(epoch, cfg_b3)
    mode = cfg_b3.get("release_mode", "constant")
    out_coeff = base
    hid_coeff = base

    if mode == "constant":
        return out_coeff, hid_coeff

    start = int(cfg_b3.get("release_start", 0))
    if epoch <= start:
        return out_coeff, hid_coeff

    if mode == "hid_first_linear":
        end_hid = int(cfg_b3.get("release_end_hid", start + 1))
        end_out = int(cfg_b3.get("release_end_out", end_hid + 1))
        hid_floor = float(cfg_b3.get("teach_hid_floor", 0.0))
        out_floor = float(cfg_b3.get("teach_out_floor", 0.2))

        if epoch >= end_hid:
            hid_coeff = hid_floor
        else:
            frac = (epoch - start) / max(1, end_hid - start)
            hid_coeff = base * (1.0 - frac) + hid_floor * frac

        if epoch >= end_out:
            out_coeff = out_floor
        else:
            frac = (epoch - start) / max(1, end_out - start)
            out_coeff = base * (1.0 - frac) + out_floor * frac
        return out_coeff, hid_coeff

    if mode == "both_linear_floor":
        end_hid = int(cfg_b3.get("release_end_hid", start + 1))
        end_out = int(cfg_b3.get("release_end_out", end_hid + 1))
        hid_floor = float(cfg_b3.get("teach_hid_floor", 0.0))
        out_floor = float(cfg_b3.get("teach_out_floor", 0.15))

        frac_h = min(max((epoch - start) / max(1, end_hid - start), 0.0), 1.0)
        frac_o = min(max((epoch - start) / max(1, end_out - start), 0.0), 1.0)
        hid_coeff = base * (1.0 - frac_h) + hid_floor * frac_h
        out_coeff = base * (1.0 - frac_o) + out_floor * frac_o
        return out_coeff, hid_coeff

    if mode == "hard_release":
        if epoch >= start:
            return 0.0, 0.0
        return out_coeff, hid_coeff

    raise ValueError(f"Unknown release_mode={mode}")


def loss_b3_next(
    params_b3, batch, P_star, cfg, cfg_b3, teacher=None, coeffs=(1.0, 1.0)
):
    dtype = cfg["dtype"]
    out_mult, hid_mult = coeffs
    lambda_spike = jnp.asarray(cfg_b3.get("lambda_spike", 0.0), dtype=dtype)
    lambda_aux_late = jnp.asarray(cfg_b3.get("lambda_aux_late", 0.0), dtype=dtype)
    lambda_teach_out = jnp.asarray(
        cfg_b3.get("lambda_teach_out", 0.0) * out_mult, dtype=dtype
    )
    lambda_teach_hid = jnp.asarray(
        cfg_b3.get("lambda_teach_hid", 0.0) * hid_mult, dtype=dtype
    )

    fwd = forward_b3_refine(params_b3, batch, P_star, cfg, cfg_b3)
    pred = fwd["pred"]
    tgt = batch["targets"].astype(dtype)

    L_task = jnp.mean(jnp.mean((pred - tgt) ** 2, axis=-1))
    L_spike = jnp.mean(fwd["S_all"])
    total = L_task + lambda_spike * L_spike

    L_teach_out = jnp.asarray(0.0, dtype=dtype)
    L_teach_hid = jnp.asarray(0.0, dtype=dtype)
    if cfg_b3.get("use_distill", False):
        if teacher is None:
            raise ValueError("Distillation requires explicit teacher outputs")
        L_teach_out = jnp.mean(jnp.mean((pred - teacher["pred"]) ** 2, axis=-1))
        total = total + lambda_teach_out * L_teach_out
        if cfg_b3.get("lambda_teach_hid", 0.0) > 0:
            L_teach_hid = jnp.mean(
                jnp.mean((fwd["tgt_repr"] - teacher["target_repr"]) ** 2, axis=-1)
            )
            total = total + lambda_teach_hid * L_teach_hid

    L_aux_late = jnp.asarray(0.0, dtype=dtype)
    if fwd["late_preds"] is not None and cfg_b3.get("lambda_aux_late", 0.0) > 0:
        tgt_rep = jnp.broadcast_to(tgt[None, :, :], fwd["late_preds"].shape)
        L_aux_late = jnp.mean(jnp.mean((fwd["late_preds"] - tgt_rep) ** 2, axis=-1))
        total = total + lambda_aux_late * L_aux_late

    return total, {
        "loss": total,
        "L_task": L_task,
        "L_spike": L_spike,
        "L_teach_out": L_teach_out,
        "L_teach_hid": L_teach_hid,
        "L_aux_late": L_aux_late,
        "spike_rate": jnp.mean(fwd["S_all"]),
        "u_abs_mean": jnp.mean(jnp.abs(fwd["U_all"])),
    }
