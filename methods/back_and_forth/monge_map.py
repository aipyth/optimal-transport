from jax import lax
from jax import numpy as jnp

from .pushforward import _central_gradient_nd

# Reuse your _central_gradient_nd as given in the prompt.

def monge_field_from_psi_nd(psi: jnp.ndarray, keep_inside: bool = True) -> jnp.ndarray:
    """
    Build the Monge map in INDEX coordinates from a scalar potential ψ
    stored on cell centers. We interpret gradients as displacements in
    physical coordinates, convert them to index units, and add them to
    the cell-center indices (i + 0.5).
    """
    shape = psi.shape
    d = psi.ndim
    dtype = psi.dtype
    n_vec = jnp.array(shape, dtype=dtype).reshape((d,) + (1,) * d)

    grad = _central_gradient_nd(psi).astype(dtype)  # (d, *shape)
    idx = jnp.indices(shape, dtype=dtype)
    centers = idx + 0.5

    F = centers + grad * n_vec
    if keep_inside:
        for ax, nk in enumerate(shape):
            F = F.at[ax].set(jnp.clip(F[ax], 0.0, nk - 1.0))
    return F


def interp_vector_field_at_positions_nd(V: jnp.ndarray,
                                        s: jnp.ndarray,
                                        keep_inside: bool = True) -> jnp.ndarray:
    """
    Multilinear 'gather' interpolation of a vector field V at positions s
    (both in INDEX coordinates).
      - V: (d, n1, n2, ..., nd)
      - s: (d, n1, n2, ..., nd)   fractional index positions at which to sample V
    Returns:
      - out: (d, n1, n2, ..., nd)  V evaluated at s via multilinear interpolation
    Boundary behavior matches adaptive_pushforward_nd: clip s to [0, n_k-1],
    clip base to [0, n_k-2], then 2^d-corner weights.
    """
    d = V.shape[0]
    shape = V.shape[1:]
    assert s.shape == (d, *shape), "s must be (d, *shape) in index coords"

    # Per-axis clip of positions into [0, n_k - 1]
    s = jnp.stack(
        [jnp.clip(s[ax], 0.0, shape[ax] - 1.0) for ax in range(d)],
        axis=0,
    )

    base = jnp.floor(s).astype(jnp.int32)  # (d, *shape)
    # Ensure room for the +1 corner on every axis
    for ax, nk in enumerate(shape):
        base_ax = jnp.clip(base[ax], 0, nk - 2)
        base = base.at[ax].set(base_ax)

    frac = s - base.astype(jnp.float32)  # (d, *shape) in [0,1)

    # Flatten everything to (d, N) where N = prod(shape)
    N = 1
    for nk in shape:
        N *= nk

    V_flat   = V.reshape(d, N)          # (d, N)
    base_flat = base.reshape(d, N)      # (d, N)
    frac_flat = frac.reshape(d, N)      # (d, N)

    # Row-major strides for flattening
    strides_py = []
    p = 1
    for k in range(len(shape)-1, -1, -1):
        strides_py.insert(0, p)
        p *= shape[k]
    strides = jnp.array(strides_py, dtype=jnp.int32).reshape(d, 1)  # (d,1)

    out_flat = jnp.zeros_like(V_flat)

    def corner_body(m, acc):
        # Bits ∈ {0,1}^d select the corner
        bits = jnp.array([(m >> k) & 1 for k in range(d)],
                         dtype=jnp.int32).reshape(d, 1)            # (d,1)
        corner_idx = base_flat + bits                               # (d,N)

        # weights: product_k [ frac_k if bit=1 else (1-frac_k) ]
        w_comp = jnp.where(bits == 1, frac_flat, 1.0 - frac_flat)    # (d,N)
        w = jnp.prod(w_comp, axis=0)                                 # (N,)

        # Linearized indices
        flat_idx = jnp.sum(corner_idx * strides, axis=0)             # (N,)

        # Gather: shape (d, N)
        vals = jnp.take(V_flat, flat_idx, axis=1)
        return acc + vals * w  # broadcast (N,) to (d,N)

    out_flat = lax.fori_loop(0, 1 << d, corner_body, out_flat)
    out = out_flat.reshape((d, *shape))

    if keep_inside:
        out = jnp.stack(
            [jnp.clip(out[ax], 0.0, shape[ax] - 1.0) for ax in range(d)],
            axis=0,
        )
    return out


def compose_monge_map_nd(T: jnp.ndarray,
                         psi: jnp.ndarray | None = None,
                         F: jnp.ndarray | None = None,
                         keep_inside: bool = True) -> jnp.ndarray:
    """
    Compose maps in INDEX coordinates with consistent clipping/interp:
      - If psi is given: build F = monge_field_from_psi_nd(psi),
        then return F ∘ T evaluated via multilinear gather.
      - Or pass a precomputed F (shape (d, *shape)).
    Shapes:
      T : (d, n1, n2, ..., nd)   -- a map in index coords (per-voxel targets)
      psi: (n1, n2, ..., nd)     -- potential (optional)
      F : (d, n1, n2, ..., nd)   -- vector field map in index coords (optional)

    Returns:
      T_new = (F ∘ T) with CIC-gather interpolation and boundary clamping.
    """
    if (psi is None) == (F is None):
        raise ValueError("Pass exactly one of {psi, F}.")

    if F is None:
        F = monge_field_from_psi_nd(psi, keep_inside=keep_inside)

    # Sanity: T and F must share the same spatial shape
    if T.shape != F.shape:
        raise ValueError(f"Shape mismatch: T{T.shape} vs F{F.shape}")

    # Evaluate F at fractional index positions given by T
    T_new = interp_vector_field_at_positions_nd(F, T, keep_inside=keep_inside)
    return T_new


# Convenience: build a fresh Monge map directly from psi (index coords)
def monge_map_from_psi_nd(psi: jnp.ndarray, keep_inside: bool = True) -> jnp.ndarray:
    """
    Return F(i)=i+grad(psi)(i)*n in INDEX coords, optionally clamped into domain.
    """
    return monge_field_from_psi_nd(psi, keep_inside=keep_inside)


# (Optional) helpers for unit/index coordinate conversion:
def index_to_unit(T: jnp.ndarray) -> jnp.ndarray:
    d = T.shape[0]
    shape = T.shape[1:]
    n_vec = jnp.array(shape, dtype=jnp.float32).reshape((d,) + (1,) * len(shape))
    return T / n_vec

def unit_to_index(U: jnp.ndarray) -> jnp.ndarray:
    d = U.shape[0]
    shape = U.shape[1:]
    n_vec = jnp.array(shape, dtype=jnp.float32).reshape((d,) + (1,) * len(shape))
    return U * n_vec
