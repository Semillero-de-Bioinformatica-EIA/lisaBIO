import torch
import torch.nn as nn

class WSIBagEncoder(nn.Module):
    """
    Encoder de aprendizaje por instancias múltiples (MIL - Multiple Instance Learning) 
    con atención gated para resumir la bolsa de parches WSI en una representación de paciente.
    """

    def __init__(self, input_dim: int = 512, output_dim: int = 128):
        super(WSIBagEncoder, self).__init__()
        self.attention_a = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh()
        )
        self.attention_b = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Sigmoid()
        )
        self.attention_c = nn.Linear(128, 1)
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N_parches, input_dim)
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = self.attention_c(a * b)  # (N_parches, 1)
        A = torch.softmax(A, dim=0)

        # Representación ponderada por atención de la lámina
        bag_representation = torch.mm(A.T, x) # (1, input_dim)
        return self.projection(bag_representation)
