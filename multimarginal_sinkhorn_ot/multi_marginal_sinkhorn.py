import torch

def sinkhorn_update(
        C1: torch.Tensor,
        C2: torch.Tensor,
        C3: torch.Tensor,
        mu1: torch.Tensor,
        mu2: torch.Tensor,
        mu3: torch.Tensor,
        epsilon: float = 0.01,
        num_iters: int = 1_000,
        ):
    n = mu1.shape.count()
    m = mu2.shape.count()
    l = mu3.shape.count()
    # u1 = torch.randn(n, requires_grad=True)
    # u2 = torch.randn(m, requires_grad=True)
    # u3 = torch.randn(l, requires_grad=True)
    u1 = torch.randn(n)
    u2 = torch.randn(m)
    u3 = torch.randn(l)
    for _ in range(num_iters):
        # Update u1
        K1 = torch.sum(
            torch.exp(
                (u2.unsqueeze(0) - C1) / epsilon
                ) * torch.exp((u3.unsqueeze(1) - C2 - C3) / epsilon),
            dim=(1, 2)
            )
        u1 = epsilon * torch.log(mu1) - epsilon * torch.log(K1)

        # Update u2
        K2 = torch.sum(
            torch.exp(
                (u1.unsqueeze(1) - C1.T) / epsilon
                ) * torch.exp((u3.unsqueeze(0) - C2.T - C3.T) / epsilon),
            dim=(0, 2)
            )
        u2 = epsilon * torch.log(mu2) - epsilon * torch.log(K2)

        # Update u3
        K3 = torch.sum(
            torch.exp(
                (u1.unsqueeze(2) - C2) / epsilon
                ) * torch.exp((u2.unsqueeze(2) - C3) / epsilon),
            dim=(0, 1)
            )
        u3 = epsilon * torch.log(mu3) - epsilon * torch.log(K3)

    # Compute π*
    pi_star = torch.exp((u1.unsqueeze(1).unsqueeze(2) + u2.unsqueeze(0).unsqueeze(2) + u3.unsqueeze(0).unsqueeze(1) - C1 - C2 - C3) / epsilon)

    return pi_star, u1, u2, u3



if __name__ == '__main__':

    n = m = l = 10

    # Initialize u vectors, cost matrices, and marginals
    u1 = torch.randn(n, requires_grad=True)
    u2 = torch.randn(m, requires_grad=True)
    u3 = torch.randn(l, requires_grad=True)

    C1 = torch.randn(n, m) # cost matrix for C1(x, y)
    C2 = torch.randn(n, l) # cost matrix for C2(x, z)
    C3 = torch.randn(m, l) # cost matrix for C3(y, z)

    mu1 = torch.randn(n)
    mu2 = torch.randn(m)
    mu3 = torch.randn(l)

    epsilon = 0.3 # regularization parameter
    num_iters = 100 # number of iterations

    # Run the algorithm
    pi_star, updated_u1, updated_u2, updated_u3 = sinkhorn_update(u1, u2, u3, C1, C2, C3, mu1, mu2, mu3, epsilon, num_iters)
