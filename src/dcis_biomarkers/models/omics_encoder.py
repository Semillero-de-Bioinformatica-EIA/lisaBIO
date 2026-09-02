"""
Residual Deep MLP Omics Encoder for High-Dimensional Genomic Vectors (METABRIC Z-scores & CNA).
Compresses transcriptomic and copy-number alteration profiles into a robust feature representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OmicsResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super(OmicsResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.ln1(self.fc1(x)))
        out = self.dropout(out)
        out = self.ln2(self.fc2(out))
        return F.relu(out + residual)


class OmicsEncoder(nn.Module):
    """
    Encoder Residual Multicapa para datos omicos de alta dimension (METABRIC 1,000-20,000 genes).
    Utiliza LayerNorm para asegurar estabilidad con cualquier tamano de batch (1..N).
    """

    def __init__(
        self,
        input_dim: int = 1000,
        hidden_dim: int = 256,
        output_dim: int = 512,
        num_res_blocks: int = 2,
        dropout: float = 0.3
    ):
        super(OmicsEncoder, self).__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        blocks = []
        for _ in range(num_res_blocks):
            blocks.append(OmicsResidualBlock(hidden_dim, dropout=dropout))
        self.res_blocks = nn.Sequential(*blocks)
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (Batch, input_dim)
        h = self.input_layer(x)
        h = self.res_blocks(h)
        return self.output_layer(h) # Output shape: (Batch, output_dim)
