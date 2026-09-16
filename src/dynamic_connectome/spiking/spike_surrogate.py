import jax


def spike_fn(U, theta, cfg_b3):
    beta = float(cfg_b3.get("surrogate_beta", 10.0))

    @jax.custom_jvp
    def _inner(x):
        return (x >= 0.0).astype(x.dtype)

    @_inner.defjvp
    def _inner_jvp(primals, tangents):
        (x,) = primals
        (x_dot,) = tangents
        y = _inner(x)
        s = jax.nn.sigmoid(beta * x)
        surrogate = beta * s * (1.0 - s)
        y_dot = surrogate * x_dot
        return y, y_dot

    return _inner(U - theta)


__all__ = ["spike_fn"]
