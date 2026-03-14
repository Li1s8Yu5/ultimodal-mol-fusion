import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Base: two-tower projection
# ---------------------------
class TwoTowerProjector(nn.Module):
    """
    Projects embedding and fingerprint to the same hidden_dim.
    """

    def __init__(self, embedding1_dim, embedding2_dim, hidden_dim):
        super().__init__()
        self.embedding_proj = nn.Linear(embedding1_dim, hidden_dim)
        self.fingerprint_proj = nn.Linear(embedding2_dim, hidden_dim)

    def forward(self, embedding, fingerprint):
        emb = self.embedding_proj(embedding)
        fp = self.fingerprint_proj(fingerprint)
        return emb, fp


class LearnableScalarAttentionFusion(nn.Module):
    """
    标量可学习注意力：softmax([alpha, beta]) 决定两路权重
    """

    def __init__(self, embedding1_dim, embedding2_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.projector = TwoTowerProjector(embedding1_dim, embedding2_dim, hidden_dim)

        self.alpha_raw = nn.Parameter(torch.tensor(0.0))
        self.beta_raw = nn.Parameter(torch.tensor(0.0))

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embedding, fingerprint):
        emb, fp = self.projector(embedding, fingerprint)

        weights = F.softmax(torch.stack([self.alpha_raw, self.beta_raw]), dim=0)
        fused = weights[0] * emb + weights[1] * fp

        fused = self.dropout(self.norm(fused))
        return fused


# ---------------------------
# Classifier with selectable fusion
# ---------------------------
class FusionMLPClassifier(nn.Module):
    def __init__(self, embedding1_dim, embedding2_dim,
                 hidden_dim=256, mlp_hidden_dim=64, dropout=0.3,
                 gate_hidden_dim=128,
                 gate_features="concat"):
        super().__init__()

        self.fusion = LearnableScalarAttentionFusion(
            embedding1_dim, embedding2_dim, hidden_dim=hidden_dim, dropout=dropout
        )

        # 可学习放缩 & 中心化参数（保留你的设计；center 你若不需要可删）
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.logit_center = nn.Parameter(torch.tensor(0.0))

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.BatchNorm1d(mlp_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, 1)
        )

    def forward(self, embedding, fingerprint):
        fused = self.fusion(embedding, fingerprint)
        logits = self.mlp(fused).squeeze(-1)

        scale = torch.clamp(self.logit_scale, 0.5, 5.0)
        logits = logits * scale

        return logits
