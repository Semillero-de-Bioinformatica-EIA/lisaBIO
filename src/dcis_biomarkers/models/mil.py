import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedAttentionMIL(nn.Module):
    """
    Gated Attention Multiple Instance Learning (ABMIL).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.25):
        super().__init__()
        self.L = input_dim
        self.D = hidden_dim
        self.K = 1

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )
        
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )
        
        self.attention_weights = nn.Linear(self.D, self.K)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, mask: torch.Tensor = None):
        """
        h: (batch_size, num_instances, input_dim)
        mask: (batch_size, num_instances) boolean mask (True = valid instance)
        """
        # A = w^T * (tanh(V * h^T) * sigmoid(U * h^T))
        A_V = self.attention_V(h)  # B x N x D
        A_U = self.attention_U(h)  # B x N x D
        A = self.attention_weights(A_V * A_U) # B x N x 1
        A = torch.transpose(A, 2, 1) # B x 1 x N
        
        if mask is not None:
            # Enmascarar instancias inválidas poniendo atención en -inf
            A = A.masked_fill(~mask.unsqueeze(1), float('-inf'))
            
        A = F.softmax(A, dim=2) # B x 1 x N
        A = self.dropout(A)
        
        # M = A * h
        M = torch.bmm(A, h) # B x 1 x input_dim
        M = M.squeeze(1) # B x input_dim
        
        return M, A.squeeze(1)
