import jax
from jax import lax
from jax import numpy as jnp


def _build_envelope(varphi_row, coords):
    """
    Build upper envelope of lines: L_i(x) = x * coords[i] - varphi_row[i],
    with coords sorted increasing. Returns hull parameters and size.
    """
    n = coords.shape[0]
    init_state = {
        'hull_m': jnp.zeros(n),         # slopes (coords)
        'hull_b': jnp.zeros(n),         # intercepts = -varphi
        'breakpoints': jnp.zeros(n),    # switch x positions between consecutive hull segments
        'size': 0
    }

    def add_line(state, idx):
        m = coords[idx]
        b = -varphi_row[idx]  # intercept
        hull_m = state['hull_m']
        hull_b = state['hull_b']
        breakpoints = state['breakpoints']
        size = state['size']

        def cond_fn(s_and_vals):
            s, hull_m, hull_b, breakpoints = s_and_vals
            # need at least two existing lines to consider popping
            prev_m = hull_m[s - 1]
            prev_b = hull_b[s - 1]
            prev2_m = hull_m[s - 2]
            prev2_b = hull_b[s - 2]
            # x-coordinate where prev2 and prev switch
            x_prev = (prev2_b - prev_b) / (prev_m - prev2_m)
            # x where prev and new intersect
            x_new = (prev_b - b) / (m - prev_m)
            return (s >= 2) & (x_new <= x_prev)

        def body_fn(s_and_vals):
            s, hull_m, hull_b, breakpoints = s_and_vals
            return (s - 1, hull_m, hull_b, breakpoints)

        # possibly pop last lines until convexity holds
        s, hull_m, hull_b, breakpoints = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (size, hull_m, hull_b, breakpoints)
        )

        # append new line at position s
        hull_m = hull_m.at[s].set(m)
        hull_b = hull_b.at[s].set(b)

        # compute new breakpoint with previous line (if exists)
        def compute_bp():
            prev_m = hull_m[s - 1]
            prev_b = hull_b[s - 1]
            return (prev_b - b) / (m - prev_m)
        new_bp = jax.lax.cond(s == 0, lambda: 0.0, compute_bp)
        breakpoints = breakpoints.at[s - 1].set(new_bp)
        # breakpoints = lax.cond(
        #     s > 0,
        #     lambda bp: bp.at[s-1].set(new_bp),
        #     lambda bp: bp,
        #     breakpoints,
        #     )

        size = s + 1
        return {'hull_m': hull_m, 'hull_b': hull_b, 'breakpoints': breakpoints, 'size': size}, None

    state_final, _ = jax.lax.scan(add_line, init_state, jnp.arange(coords.shape[0]))
    return state_final['hull_m'], state_final['hull_b'], state_final['breakpoints'], state_final['size']


def _evaluate_envelope(hull_m, hull_b, breakpoints, hull_size, xs):
    """
    Given built hull, evaluate envelope at sorted xs.
    """
    def scan_fn(ptr, x):
        # advance pointer while next segment dominates
        def cond(p):
            return (p + 1 < hull_size) & (x >= breakpoints[p])
        def body(p):
            return p + 1
        ptr = jax.lax.while_loop(cond, body, ptr)
        val = x * hull_m[ptr] + hull_b[ptr]
        return ptr, val

    _, vals = jax.lax.scan(scan_fn, 0, xs)
    return vals


@jax.jit
def legendre_1d_fast_row(varphi_row, coords):
    """
    Fast 1D Legendre transform (convex conjugate) for one vector.
    varphi_row: shape (n,)
    coords: shape (n,) sorted increasing
    returns: varphi_star of shape (n,)
    """
    hull_m, hull_b, breakpoints, hull_size = _build_envelope(varphi_row, coords)
    xs = coords  # evaluate at same grid
    conj = _evaluate_envelope(hull_m, hull_b, breakpoints, hull_size, xs)
    return conj  # (n,)


def fast_legendre_conjugate_along_axis(varphi, coords, axis):
    """
    Applies linear-time Legendre conjugation along specified axis of varphi.
    varphi: array of shape (n1, n2, ..., nd)
    coords: 1D array for that axis (length matches varphi.shape[axis])
    axis: integer axis to process
    """
    moved = jnp.moveaxis(varphi, axis, -1)  # bring target axis last: (..., n)
    flat_shape = (-1, moved.shape[-1])
    flattened = moved.reshape(flat_shape)    # (batch, n)
    # vmap over batch dimension
    conj_flat = jax.vmap(legendre_1d_fast_row, in_axes=(0, None))(flattened, coords)  # (batch, n)
    conj = conj_flat.reshape(moved.shape)
    return jnp.moveaxis(conj, -1, axis)


def c_transform_quadratic_fast(phi, coords_list):
    """
    Quadratic c-transform using fast 1D Legendre per axis.
    phi: array shape (n1, ..., nd)
    coords_list: list of 1D coordinate arrays per axis
    """
    # 1) Build 0.5*||x||^2 on the grid
    half_sq = 0.0
    for axis, coords in enumerate(coords_list):
        shape = [1]*phi.ndim
        shape[axis] = coords.shape[0]
        half_sq = half_sq + (0.5 * coords**2).reshape(shape)

    # 2) Work with varphi = 0.5||x||^2 - phi  (fixed throughout!)
    varphi = half_sq - phi
    H = -varphi

    # 3) Take partial conjugates successively; this yields varphi^*
    for axis, coords in enumerate(coords_list):
        H = fast_legendre_conjugate_along_axis(-H, coords, axis)

    # 4) Convert back to psi = 0.5||y||^2 - varphi^*
    psi = half_sq - H
    return psi
