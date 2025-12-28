import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConv(nn.Module):
    """1D temporal convolution for processing frame sequences"""
    def __init__(self, channels, kernel_size=3):
        super(TemporalConv, self).__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        x = self.relu(self.bn(self.conv(x)))
        return x.transpose(1, 2)  # (B, T, C)


class TemporalAttention(nn.Module):
    """Self-attention mechanism for temporal modeling"""
    def __init__(self, channels, num_heads=8):
        super(TemporalAttention, self).__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C) tensor
        Returns:
            (B, T, C) tensor
        """
        B, T, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        
        return self.norm(out + x)  # Residual connection


class TemporalEncoder(nn.Module):
    """
    Encodes temporal information across video frames.
    
    Takes frame-level features and models temporal dependencies
    using a combination of temporal convolutions and self-attention.
    
    Args:
        channels (int): Number of feature channels
        num_layers (int): Number of temporal encoding layers
        num_heads (int): Number of attention heads
        max_seq_len (int): Maximum sequence length for positional encoding
    """
    def __init__(self, channels=256, num_layers=3, num_heads=8, max_seq_len=32):
        super(TemporalEncoder, self).__init__()
        
        self.channels = channels
        self.max_seq_len = max_seq_len
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, channels))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        
        # Temporal encoding layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'temporal_conv': TemporalConv(channels, kernel_size=3),
                'attention': TemporalAttention(channels, num_heads),
                'ffn': nn.Sequential(
                    nn.Linear(channels, channels * 4),
                    nn.GELU(),
                    nn.Linear(channels * 4, channels),
                    nn.LayerNorm(channels)
                )
            }) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(channels)
    
    def forward(self, x, frame_mask=None):
        """
        Args:
            x: Temporal features of shape (B, T, C)
            frame_mask: Optional mask for variable-length sequences (B, T)
        
        Returns:
            Encoded temporal features (B, T, C)
        """
        B, T, C = x.shape
        
        # Add positional encoding
        if T <= self.max_seq_len:
            x = x + self.pos_embedding[:, :T, :]
        else:
            # Interpolate positional encoding for longer sequences
            pos = F.interpolate(
                self.pos_embedding.transpose(1, 2),
                size=T,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            x = x + pos
        
        # Apply temporal encoding layers
        for layer in self.layers:
            # Temporal convolution
            x = layer['temporal_conv'](x)
            
            # Self-attention
            x = layer['attention'](x)
            
            # Feed-forward network
            residual = x
            x = layer['ffn'](x)
            x = x + residual
        
        return self.norm(x)


if __name__ == "__main__":
    # Test the temporal encoder
    model = TemporalEncoder(channels=256, num_layers=3, num_heads=8)
    
    # Simulate temporal features (batch_size=2, time_steps=16, channels=256)
    x = torch.randn(2, 16, 256)
    
    output = model(x)
    print("Temporal Encoder Test:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")