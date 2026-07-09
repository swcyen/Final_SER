import torch
import torch.nn as nn
import torch.nn.functional as F


class SSLModel(nn.Module):

    def __init__(self, encoder):

        super().__init__()

        self.encoder = encoder

        # Two-layer projector MLP (hidden -> 512 -> proj_dim)
        hidden = max(512, encoder.embedding_dim)
        proj_dim = 128
        self.projector = nn.Sequential(
            nn.Linear(encoder.embedding_dim, hidden),
            nn.GELU(),
            nn.GroupNorm(8, hidden),
            nn.Linear(hidden, proj_dim),
        )

    def forward(self, waveform):

        embedding = self.encoder(waveform)

        projection = self.projector(embedding)

        projection = F.normalize(
            projection,
            dim=-1
        )

        return projection