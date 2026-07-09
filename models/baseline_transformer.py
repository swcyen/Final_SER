import torch
import torch.nn as nn
from typing import List, Optional


class BaselineSERTransformer(nn.Module):
    """Raw-waveform CNN + Transformer baseline for speech emotion recognition."""

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
        classifier_dropout: float = 0.3,
        pooling: str = "mean",
        cnn_channels: Optional[List[int]] = None,
        cnn_kernels: Optional[List[int]] = None,
        cnn_strides: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.pooling = pooling

        # Default CNN config
        if cnn_channels is None:
            cnn_channels = [64, 128, 256]
        if cnn_kernels is None:
            cnn_kernels = [10, 8, 4]
        if cnn_strides is None:
            cnn_strides = [5, 4, 2]

        # Build CNN dynamically from config
        cnn_layers = [
            nn.Conv1d(1, cnn_channels[0], kernel_size=cnn_kernels[0], stride=cnn_strides[0], padding=2),
            nn.BatchNorm1d(cnn_channels[0]),
            nn.GELU(),
        ]
        
        for i in range(1, len(cnn_channels)):
            cnn_layers.extend([
                nn.Conv1d(cnn_channels[i-1], cnn_channels[i], kernel_size=cnn_kernels[i], stride=cnn_strides[i], padding=2),
                nn.BatchNorm1d(cnn_channels[i]),
                nn.GELU(),
            ])
        
        self.cnn = nn.Sequential(*cnn_layers)
        # Feature dimension produced by CNN (last channel)
        feature_dim = cnn_channels[-1]

        # If CNN feature dim differs from transformer hidden dim, project features.
        if feature_dim != hidden_dim:
            self.projection = nn.Linear(feature_dim, hidden_dim)
        else:
            self.projection = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            waveform: Raw audio tensor with shape [batch, 1, audio_samples].

        Returns:
            Emotion logits with shape [batch, num_classes].
        """
        features = self.cnn(waveform)  # [batch, feature_dim, sequence_length]
        features = features.transpose(1, 2)  # [batch, sequence_length, feature_dim]

        # Project CNN features to transformer hidden dim if necessary
        if self.projection is not None:
            features = self.projection(features)  # [batch, sequence_length, hidden_dim]

        contextual = self.transformer(features)  # [batch, sequence_length, hidden_dim]
        if self.pooling == "mean":
            utterance_embedding = contextual.mean(dim=1)  # [batch, 256]
        elif self.pooling == "max":
            utterance_embedding = contextual.max(dim=1)[0]  # [batch, 256]
        else:
            utterance_embedding = contextual[:, 0, :]  # [batch, 256] - CLS token style
        logits = self.classifier(utterance_embedding)  # [batch, num_classes]
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
