import torch
from typing import Callable


def sinkhorn(
        C: Callable,
        p: torch.Tensor,
        q: torch.Tensor,
        epsilon: float,
        max_iters: int = 1000,
        stop_threshold: float = 1e-9,
        ):
    """
    Sinkhorn algorithm for 3-marginal regularized optimal transport.

    Args:
        C (torch.Tensor): Cost matrix of shape (N, M, K).
        p (torch.Tensor): Source distribution of shape (N, 1).
        q (torch.Tensor): Target distribution of shape (M, 1).
        s (torch.Tensor): Target distribution of shape (K, 1).
        epsilon (float): Regularization parameter.
        max_iters (int): Maximum number of iterations.
        stop_threshold (float): Convergence threshold.

    Returns:
        torch.Tensor: Regularized transport plan of shape (N, M, K).
    """

    # Compute the kernel matrix
    K = torch.exp(-C / epsilon)
    K_transpose = K.t()

    # Initialize scaling vectors
    a = torch.ones_like(p)

    # Sinkhorn iterations
    for _ in range(max_iters):
        b = q / (K_transpose @ a)
        a = p / (K @ b)

        # Check for convergence
        # if torch.max(torch.abs(a - prev_a)) < stop_threshold:
        #     break
        # prev_a = a

    # Compute the transport plan
    a_expanded = a.expand_as(K)
    b_expanded = b.t().expand_as(K)
    P = a_expanded * K * b_expanded
    
    return P
