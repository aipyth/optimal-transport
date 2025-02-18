import torch
from typing import Callable


def generate_sinkhorn_problem(
        source: torch.distributions.Distribution,
        target: torch.distributions.Distribution,
        cost: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        num_points: int = 10000,
        ):
    """
    Generates a random problem instance for the Sinkhorn algorithm
    using the two distributions for the source and target marginals
    on the [0,1] space.

    Args:
        source (torch.distributions.Distribution): source distribution
        target (torch.distributions.Distribution): target distribution

    Returns:
        C (torch.Tensor): Cost matrix of shape (num_sources, num_targets).
        p (torch.Tensor): Source distribution of shape (num_sources, 1).
        q (torch.Tensor): Target distribution of shape (num_targets, 1).
    """
    source_points = torch.linspace(0, 1, steps=num_points).unsqueeze(1)
    target_points = torch.linspace(0, 1, steps=num_points).unsqueeze(1)

    C = cost(source_points, target_points)

    p = torch.exp(source.log_prob(source_points))
    p = p / p.sum()
    q = torch.exp(target.log_prob(target_points))
    q = q / q.sum()

    return C, p, q, source_points, target_points


if __name__ == '__main__':
    # Usage example
    C, p, q, source_points, target_points = generate_sinkhorn_problem(
        torch.distributions.Normal(0.3, 0.04),
        torch.distributions.Normal(0.8, 0.1),
        cost=lambda x, y: torch.cdist(x, y, p=1),
        num_points=300
    )

    print("Cost matrix C:\n", C)
    print("Source distribution p:\n", p)
    print("Target distribution q:\n", q)
