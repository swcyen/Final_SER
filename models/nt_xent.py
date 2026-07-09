import torch
import torch.nn as nn


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).

    Expects two batches of embeddings `z_i`, `z_j` with shape `(N, dim)`.
    Returns scalar loss (mean over 2N views).
    """

    def __init__(self, temperature: float = 0.5, eps: float = 1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        if z_i.dim() != 2 or z_j.dim() != 2:
            raise ValueError("z_i and z_j must be 2D tensors of shape (N, dim)")
        if z_i.shape != z_j.shape:
            raise ValueError("z_i and z_j must have the same shape")

        device = z_i.device
        N = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)  # 2N x dim
        z = nn.functional.normalize(z, dim=1)

        # similarity matrix
        sim = torch.matmul(z, z.T) / (self.temperature + self.eps)  # 2N x 2N

        # mask to zero out self-similarities
        diag_mask = torch.eye(2 * N, device=device, dtype=torch.bool)

        # for each i, positive index is i+N (mod 2N)
        pos_idx = (torch.arange(2 * N, device=device) + N) % (2 * N)

        # exponentiate similarities (with numerical stability)
        # subtract max per row for stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim_exp = torch.exp(sim - sim_max)
        sim_exp = sim_exp * (~diag_mask).float()

        denom = sim_exp.sum(dim=1)
        # numerator is exp(sim[i, positive])
        numer = torch.exp(sim[torch.arange(2 * N, device=device), pos_idx] - sim_max.squeeze(1))

        loss = -torch.log((numer / (denom + self.eps)) + self.eps)
        return loss.mean()


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5):
    loss_module = NTXentLoss(temperature=temperature)
    return loss_module(z1, z2)
