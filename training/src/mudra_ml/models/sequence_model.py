"""Sequence classifiers built from: per-frame encoder -> temporal model -> classifier.

    temporal = "none"         Landmark MLP + mean pooling (ablation B, no temporal modelling)
    temporal = "bilstm"       Landmark MLP + BiLSTM (ablation C; spec's main architecture)
    temporal = "transformer"  Landmark MLP + Transformer encoder (comparison)

Logits are returned (no softmax) for use with CrossEntropyLoss.

    head = "linear"   LayerNorm -> Dropout -> Linear (all models up to v8)
    head = "cosine"   additive-margin cosine head (CosFace): logits = s * cos(angle to the class), and during
                      training the true class's cosine is reduced by m. Every word must then win by a margin,
                      which pushes words apart in the embedding space (the "barrier" between words); the same
                      normalised embeddings are what custom-word prototypes are matched against.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from mudra_ml.models.landmark_mlp import LandmarkMLP


@dataclass
class ModelConfig:
    input_dim: int
    num_classes: int
    temporal: str = "bilstm"
    mlp_hidden: int = 256
    mlp_out: int = 128
    lstm_hidden: int = 256
    lstm_layers: int = 2
    tf_dim: int = 192
    tf_layers: int = 4
    tf_heads: int = 4
    tf_ff: int = 384
    max_len: int = 128
    dropout: float = 0.3
    head: str = "linear"
    cos_scale: float = 30.0
    cos_margin: float = 0.25

    @classmethod
    def from_dict(cls, values: dict) -> ModelConfig:
        values = dict(values)
        values.pop("appearance_dim", None)  # older model cards; the RGB branch was never used
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)


class AttentionPool(nn.Module):
    """Learned weighted average over time (handles signs whose key moment is not the last frame)."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x).squeeze(-1), dim=1)
        return torch.einsum("bt,btd->bd", weights, x)


class CosineMarginHead(nn.Module):
    def __init__(self, dim: int, num_classes: int, scale: float, margin: float, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.empty(num_classes, dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale, self.margin = scale, margin

    def features(self, z: torch.Tensor) -> torch.Tensor:
        """The embedding the classes are compared with (unit length)."""
        return nn.functional.normalize(self.norm(z), dim=-1)

    def forward(self, z: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        f = nn.functional.normalize(self.drop(self.norm(z)), dim=-1)
        cos = f @ nn.functional.normalize(self.weight, dim=-1).T
        if labels is not None and self.margin:
            cos = cos - self.margin * nn.functional.one_hot(labels, cos.shape[1]).to(cos.dtype)
        return self.scale * cos


class SequenceClassifier(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.frame_encoder = LandmarkMLP(cfg.input_dim, cfg.mlp_hidden, cfg.mlp_out, dropout=cfg.dropout / 2)
        fused_dim = cfg.mlp_out

        if cfg.temporal == "none":
            self.temporal = None
            out_dim = fused_dim
        elif cfg.temporal == "bilstm":
            self.temporal = nn.LSTM(
                fused_dim,
                cfg.lstm_hidden,
                num_layers=cfg.lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
            )
            out_dim = cfg.lstm_hidden * 2
        elif cfg.temporal == "transformer":
            self.proj = nn.Linear(fused_dim, cfg.tf_dim)
            self.pos = nn.Parameter(torch.zeros(1, cfg.max_len, cfg.tf_dim))
            nn.init.trunc_normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(
                cfg.tf_dim, cfg.tf_heads, cfg.tf_ff, dropout=cfg.dropout, batch_first=True, norm_first=True
            )
            self.temporal = nn.TransformerEncoder(layer, cfg.tf_layers, enable_nested_tensor=False)
            out_dim = cfg.tf_dim
        else:
            raise ValueError(f"Unknown temporal model {cfg.temporal!r}")

        self.pool = AttentionPool(out_dim) if self.temporal is not None else None
        if cfg.head == "cosine":
            self.head = CosineMarginHead(out_dim, cfg.num_classes, cfg.cos_scale, cfg.cos_margin, cfg.dropout)
        elif cfg.head == "linear":
            self.head = nn.Sequential(nn.LayerNorm(out_dim), nn.Dropout(cfg.dropout), nn.Linear(out_dim, cfg.num_classes))
        else:
            raise ValueError(f"Unknown head {cfg.head!r}")

    def encode(self, landmarks: torch.Tensor) -> torch.Tensor:
        """[B, T, D] -> pooled sequence representation [B, out_dim]."""
        x = self.frame_encoder(landmarks)

        if self.cfg.temporal == "none":
            return x.mean(dim=1)
        if self.cfg.temporal == "bilstm":
            x, _ = self.temporal(x)
        else:
            x = self.proj(x) + self.pos[:, : x.shape[1]]
            x = self.temporal(x)
        return self.pool(x)

    def embedding_from(self, z: torch.Tensor) -> torch.Tensor:
        """Embedding used for custom-word prototypes: the cosine head's unit features, else the pooled output."""
        return self.head.features(z) if isinstance(self.head, CosineMarginHead) else z

    def forward(self, landmarks: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        """labels (training only) apply the cosine head's margin; ignored by the linear head."""
        z = self.encode(landmarks)
        if isinstance(self.head, CosineMarginHead):
            return self.head(z, labels)
        return self.head(z)


def build_model(cfg: ModelConfig) -> SequenceClassifier:
    return SequenceClassifier(cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
