import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    Absolute positional embedding for Transformer
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # [max_len, d_model] -> [1, max_len, d_model] (add batch dimension)
        self.pe = pe.unsqueeze(0)
        self.register_buffer('pe_buffer', self.pe)

    def forward(self, x):
        """
        x: [B, T, D_model]
        """
        seq_len = x.size(1)  # T
        # Positional embeddings are pre-computed to be large enough for any seq_len, then sliced
        x = x + self.pe_buffer[:, :seq_len, :].to(x.device)
        return x


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding, similar to the one used in "Attention is All You Need".
    """
    def __init__(self, time_embed_dim):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        half_dim = self.time_embed_dim // 2
        # Precompute the frequency term
        # div_term = exp(log(10000) * (-2i / D))
        div_term = torch.exp(torch.arange(half_dim, dtype=torch.float32) * (-math.log(10000.0) / half_dim))
        self.register_buffer('div_term', div_term)
        
        # MLP to further process the embedding
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )

    def forward(self, t):
        """
        t: [B] (scalar timestep for each sample)
        """
        # [B] -> [B, 1] -> [B, half_dim]
        freqs = t.float().unsqueeze(1) * self.div_term.unsqueeze(0)
        
        # [B, half_dim] for sin and [B, half_dim] for cos are concatenated
        # -> [B, time_embed_dim]
        sinusoidal_emb = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)
        
        # Pass through MLP
        emb = self.mlp(sinusoidal_emb)
        return emb

    

class ConditionalTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, time_embed_dim):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, batch_first=True)
        # Linear layer to receive the timestep embedding and generate scale and shift
        self.time_fc = nn.Linear(time_embed_dim, d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, t_emb=None):
        """
        src: [B, T, d_model]
        t_emb: [B, time_embed_dim]
        """

        t_emb = self.time_fc(t_emb)  # [B, d_model]
        # Proceed with the regular Transformer Encoder Layer process
        return super().forward(src+t_emb.unsqueeze(1), src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)


class TransformerModel(nn.Module):
    def __init__(
        self,
        in_channels=2,        # Number of channels for input x
        out_channels=2,       # Number of output channels
        d_model=256,          # Transformer embedding dimension
        nhead=4,              # Number of heads for Multi-Head Attention
        num_layers=6,         # Number of Transformer layers
        dim_feedforward=1024, # Intermediate dimension of the FFN
        dropout=0.1,
        max_len=500,
    ):
        super().__init__()
        self.time_embed_dim = d_model

        self.time_embed = SinusoidalTimeEmbedding(
            time_embed_dim=self.time_embed_dim,
        )

        self.input_linear = nn.Linear(in_channels, d_model)

        self.start_proj = nn.Linear(2, d_model)
        self.goal_proj = nn.Linear(2, d_model)

        # Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)

        # Transformer Encoder
        self.layers = nn.ModuleList([
            ConditionalTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="relu",
                time_embed_dim=self.time_embed_dim)
            for _ in range(num_layers)
        ])

        self.output_linear = nn.Linear(d_model, out_channels)

    def forward(self, x, timesteps, start, goal):
        """
        x:     [B, in_channels, T]
        start: [B, 2]
        goal:  [B, 2]
        timesteps: [B] (for scalar timestep embedding)
        Returns: [B, out_channels, T]
        """
        x = x.permute(0, 2, 1)  # [B, T, in_channels]

        # Embed each timestep of the trajectory
        h = self.input_linear(x)  # [B, T, d_model]
        h = self.pos_encoding(h)

        t_emb = self.time_embed(timesteps)  # [B, time_embed_dim]

        start_token = self.start_proj(start).unsqueeze(1)  # [B, 1, d_model]
        goal_token = self.goal_proj(goal).unsqueeze(1)    # [B, 1, d_model]
        h = torch.cat([start_token, h, goal_token], dim=1)  # [B, T+2, d_model]

        for layer in self.layers:
            h = layer(h, t_emb=t_emb)

        h = h[:, 1:-1, :]  # [B, T, d_model]
        out = self.output_linear(h)
        out = out.permute(0, 2, 1)  # [B, out_channels, T]
        return out
