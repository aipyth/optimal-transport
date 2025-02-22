import matplotlib.pyplot as plt
import jax.numpy as jnp


def plot_ot_solution(x, pdf1, pdf2, transport_plan, title="Optimal Transport Solution"):
    """
    Plots the optimal transport solution with aligned marginals.

    Parameters:
    - x: Array of support points (same for both measures).
    - pdf1: First probability density function.
    - pdf2: Second probability density function.
    - transport_plan: Computed optimal transport plan (matrix).
    - title: Title of the figure.
    """
    fig = plt.figure(figsize=(6, 6))
    fig.suptitle(title, fontsize=12)
    gs = fig.add_gridspec(
        2, 2, width_ratios=[2, 6], height_ratios=[2, 6], wspace=0.05, hspace=0.1
    )
    # grid = plt.GridSpec(3, 3, wspace=0.1, hspace=0.1)

    # main transport plan visualization
    ax_main = fig.add_subplot(gs[1, 1])
    im = ax_main.imshow(transport_plan, origin="lower")
    ax_main.set_xticks([])
    ax_main.set_yticks([])

    # top plot (pdf2)
    ax_top = fig.add_subplot(gs[0, 1])
    ax_top.plot(x, pdf2)
    ax_top.set_xticks([])
    # ax_top.set_yticks([])
    # ax_top.set_xlim([x.min(), x.max()])
    # ax_top.set_title(title, fontsize=10)

    # side plot (pdf1)
    ax_side = fig.add_subplot(gs[1, 0])
    ax_side.plot(pdf1, x)
    ax_side.set_xticks([])
    # ax_side.set_yticks([])
    # ax_side.set_ylim([x.min(), x.max()])
    ax_side.invert_xaxis()  # Align with imshow

    # add colorbar
    # cbar = plt.colorbar(im, ax=ax_main, fraction=0.046)
    # cbar = plt.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04)
    # cbar.ax.tick_params(labelsize=8)

    plt.show()
