import itertools

from jax import lax
from jax import numpy as jnp


def _central_gradient_nd(psi):
    """
    Central differences with one-sided differences at boundaries.
    Returns grad with shape (d, *psi.shape).
    Assumes grid spacing h_i = 1 / n_i on [0,1]^d.
    """
    shape = psi.shape
    d = psi.ndim
    # pad ALL axes by 1 so slicing is consistent
    # but we will also handle the boundaries separately
    psi_pad = jnp.pad(psi, [(1, 1)] * d, mode='edge')

    grads = []
    for ax in range(d):
        # base slices to pick the central block for non-differenced axes
        center = [slice(1, 1 + n) for n in shape]

        # forward/backward slices along axis ax
        sl_fwd = list(center); sl_fwd[ax] = slice(2, 2 + shape[ax])   # +1 shift
        sl_bwd = list(center); sl_bwd[ax] = slice(0, shape[ax])       # -1 shift

        forward  = psi_pad[tuple(sl_fwd)]
        backward = psi_pad[tuple(sl_bwd)]
        h = 1.0 / shape[ax]           # grid step on [0,1]
        central_diff = (forward - backward) / (2.0 * h)

        # now handle boundaries with one-sided differences
        grad_ax = central_diff
        # left boundary (i=0) with forward difference
        sl_left_fwd = [slice(0, n) if i != ax else slice(1, 2) for i, n in enumerate(shape)]
        sl_left_self = [slice(0, n) if i != ax else slice(0, 1) for i, n in enumerate(shape)]
        left_fwd = psi[tuple(sl_left_fwd)]
        left_self = psi[tuple(sl_left_self)]
        left_diff = (left_fwd - left_self) / h

        # right boundary (i=n-1) with backward difference
        sl_right_bwd = [slice(0, n) if i != ax else slice(-2, -1) for i, n in enumerate(shape)]
        sl_right_self = [slice(0, n) if i != ax else slice(-1, None) for i, n in enumerate(shape)]
        right_bwd = psi[tuple(sl_right_bwd)]
        right_self = psi[tuple(sl_right_self)]
        right_diff = (right_self - right_bwd) / h

        # combine
        sl_left_assign = [slice(None)] * d; sl_left_assign[ax] = 0
        sl_right_assign = [slice(None)] * d; sl_right_assign[ax] = -1
        grad_ax = grad_ax.at[tuple(sl_left_assign)].set(left_diff.squeeze())
        grad_ax = grad_ax.at[tuple(sl_right_assign)].set(right_diff.squeeze())
        grads.append(grad_ax)

    return jnp.stack(grads, axis=0)   # (d, *shape)


def adaptive_pushforward_nd(density, psi):
    """
    Vertex-based pushforward with adaptive sub-cell sampling.
    Supports 1D/2D/3D uniform grids on [0,1]^d.
    Returns (new_density, grad_psi) to match backnforth_sqeuclidean_nd signature.
    """
    shape = density.shape
    assert psi.shape == shape
    d = density.ndim
    dtype = density.dtype

    corners = _binary_corners(d)
    num_corners = corners.shape[0]

    vertex_shape = tuple(n + 1 for n in shape)
    coords = jnp.indices(vertex_shape, dtype=dtype)
    reshape_pattern = (d,) + (1,) * d
    n_vec_vertex = jnp.array(shape, dtype=dtype).reshape(reshape_pattern)
    unit_coords = coords / n_vec_vertex

    # --- construct vertex map via centered differences evaluated by multilinear interpolation ---
    vertex_components = []
    for ax in range(d):
        step = 1.0 / shape[ax]
        shifted_plus = unit_coords.at[ax].set(unit_coords[ax] + step)
        shifted_minus = unit_coords.at[ax].set(unit_coords[ax] - step)
        f_plus = _multilinear_interpolate_cell_center(psi, shifted_plus, corners)
        f_minus = _multilinear_interpolate_cell_center(psi, shifted_minus, corners)
        grad = (f_plus - f_minus) / (2.0 * step)
        coord_axis = unit_coords[ax]
        target_coord = coord_axis + grad
        vertex_components.append(jnp.clip(target_coord, 0.0, 1.0))
    vertex_map = jnp.stack(vertex_components, axis=0)  # (d, *vertex_shape)

    # --- per-axis stretch heuristics (max edge length of vertex map over each cell) ---
    def reduce_other_axes(arr, excluded_axis):
        out = arr
        for axis in range(arr.ndim):
            if axis == excluded_axis:
                continue
            out = _adjacent_max(out, axis=axis)
        return out

    stretch_components = [
        reduce_other_axes(jnp.abs(jnp.diff(vertex_map[ax], axis=ax)), ax)
        for ax in range(d)
    ]

    slice_sizes = (2,) * d
    total_cells = density.size
    n_vec_float = jnp.array(shape, dtype=dtype)
    n_vec_int = jnp.array(shape, dtype=jnp.int32)

    def unravel(index):
        idx = index
        coords_rev = []
        for size in reversed(shape):
            size_val = jnp.int32(size)
            coords_rev.append(lax.rem(idx, size_val))
            idx = lax.div(idx, size_val)
        return tuple(reversed(coords_rev))

    def extract_cell_vertices(idx_tuple):
        starts = list(idx_tuple)
        vertices = []
        for ax in range(d):
            block = lax.dynamic_slice(vertex_map[ax], starts, slice_sizes)
            vertices.append(block.reshape(-1))
        return jnp.stack(vertices, axis=1)  # (num_corners, d)

    def evaluate_cell_map(cell_vertices, bary):
        bary = jnp.clip(bary, 0.0, 1.0)
        weights = jnp.prod(
            jnp.where(corners == 1, bary, 1.0 - bary),
            axis=1,
        )
        return jnp.sum(weights[:, None] * cell_vertices, axis=0)

    def scatter_sample(rho_array, point, sample_mass):
        scaled = point * n_vec_float - 0.5
        base = jnp.floor(scaled).astype(jnp.int32)
        base = jnp.clip(base, 0, n_vec_int - 1)
        frac = scaled - base.astype(dtype)
        other = base + jnp.sign(frac).astype(jnp.int32)
        other = jnp.clip(other, 0, n_vec_int - 1)
        frac_abs = jnp.abs(frac)
        base_w = 1.0 - frac_abs
        other_w = frac_abs

        def corner_body(m, acc):
            corner = corners[m]
            idx = jnp.where(corner == 1, other, base)
            weight_components = jnp.where(corner == 1, other_w, base_w)
            weight = jnp.prod(weight_components)
            return acc.at[tuple(idx)].add(sample_mass * weight)

        return lax.fori_loop(0, num_corners, corner_body, rho_array)

    def sample_indices(sample_idx, counts_tuple):
        idx = sample_idx
        coords_list = []
        for count in counts_tuple:
            coords_list.append(lax.rem(idx, count))
            idx = lax.div(idx, count)
        return jnp.stack(coords_list, axis=0)

    def process_cell(idx_tuple, mass, rho_array):
        cell_vertices = extract_cell_vertices(idx_tuple)
        counts_list = []
        for ax in range(d):
            stretch_val = stretch_components[ax][idx_tuple]
            est = jnp.ceil(stretch_val * shape[ax]).astype(jnp.int32)
            counts_list.append(jnp.maximum(1, est))
        counts_tuple = tuple(counts_list)
        counts_array = jnp.stack(counts_tuple)
        sample_total = jnp.prod(counts_array)
        sample_total = jnp.maximum(sample_total, 1)
        mass_per_sample = mass / sample_total.astype(dtype)
        counts_float = counts_array.astype(dtype)

        def cond_fn(state):
            return state[0] < sample_total

        def body_fn(state):
            s_idx, rho_curr = state
            local_indices = sample_indices(s_idx, counts_tuple)
            bary = (local_indices.astype(dtype) + 0.5) / counts_float
            point = evaluate_cell_map(cell_vertices, bary)
            rho_next = scatter_sample(rho_curr, point, mass_per_sample)
            return (s_idx + 1, rho_next)

        _, rho_out = lax.while_loop(cond_fn, body_fn, (jnp.int32(0), rho_array))
        return rho_out

    def cell_body(cell_flat_idx, rho_array):
        idx_tuple = unravel(jnp.int32(cell_flat_idx))
        mass = density[idx_tuple]

        def skip_fn(_):
            return rho_array

        def process_fn(_):
            return process_cell(idx_tuple, mass, rho_array)

        return lax.cond(mass <= 0, skip_fn, process_fn, operand=None)

    rho0 = jnp.zeros_like(density)
    rho = lax.fori_loop(0, total_cells, cell_body, rho0)

    target_mean = jnp.mean(density)
    rho_mean = jnp.mean(rho)
    rho = lax.cond(
        rho_mean > 0,
        lambda _: rho * (target_mean / rho_mean),
        lambda _: rho,
        operand=None,
    )

    grad = _central_gradient_nd(psi)
    return rho, grad


def _binary_corners(d):
    """
    Return the {0,1}^d corner offsets as an int32 array of shape (2^d, d).
    """
    combos = list(itertools.product((0, 1), repeat=d))
    return jnp.array(combos, dtype=jnp.int32)


def _multilinear_interpolate_cell_center(field, positions_unit, corners):
    """
    Multilinear interpolation of a cell-centered scalar field at arbitrary
    positions expressed in unit coordinates on [0,1]^d.
    `positions_unit` has shape (d, *spatial_shape); returns array of shape
    `spatial_shape`.
    """
    shape = field.shape
    d = field.ndim
    assert positions_unit.shape[0] == d
    dtype = field.dtype

    rest_shape = positions_unit.shape[1:]
    total = 1
    for n in rest_shape:
        total *= n

    pos = jnp.clip(positions_unit, 0.0, 1.0)
    reshape_pattern = (d,) + (1,) * len(rest_shape)
    n_vec = jnp.array(shape, dtype=dtype).reshape(reshape_pattern)
    limits = jnp.array(shape, dtype=jnp.int32).reshape(reshape_pattern)

    X = pos * n_vec - 0.5
    base = jnp.floor(X).astype(jnp.int32)
    base = jnp.clip(base, 0, limits - 1)
    frac = X - base.astype(dtype)
    other = base + jnp.sign(frac).astype(jnp.int32)
    other = jnp.clip(other, 0, limits - 1)

    frac_abs = jnp.abs(frac)
    base_flat = base.reshape(d, total)
    other_flat = other.reshape(d, total)
    base_w = (1.0 - frac_abs).reshape(d, total)
    other_w = frac_abs.reshape(d, total)

    field_flat = field.reshape(-1)
    strides = []
    stride = 1
    for size in reversed(shape):
        strides.insert(0, stride)
        stride *= size
    strides = jnp.array(strides, dtype=jnp.int32).reshape(d, 1)

    def corner_body(m, acc):
        bits = corners[m].reshape(d, 1)
        corner_idx = jnp.where(bits == 1, other_flat, base_flat)
        weights_comp = jnp.where(bits == 1, other_w, base_w)
        weights = jnp.prod(weights_comp, axis=0)
        flat_idx = jnp.sum(corner_idx * strides, axis=0)
        values = jnp.take(field_flat, flat_idx, axis=0)
        return acc + values * weights

    accum = lax.fori_loop(
        0,
        corners.shape[0],
        corner_body,
        jnp.zeros((total,), dtype=dtype),
    )
    return accum.reshape(rest_shape)


def _adjacent_max(arr, axis):
    """
    Max of adjacent pairs along `axis`, reducing its length by 1.
    """
    upper = arr.shape[axis]
    return jnp.maximum(
        lax.slice_in_dim(arr, 0, upper - 1, axis=axis),
        lax.slice_in_dim(arr, 1, upper, axis=axis),
    )
