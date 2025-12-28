import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module - learns which feature channels are important
    """
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W) with channel attention applied
        """
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module - learns which spatial locations are important
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, 1, H, W) spatial attention map
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    Combines channel and spatial attention sequentially
    
    Reference: "CBAM: Convolutional Block Attention Module" (Woo et al., ECCV 2018)
    
    Args:
        channels (int): Number of input channels
        reduction (int): Reduction ratio for channel attention
        kernel_size (int): Kernel size for spatial attention
    """
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W) with attention applied
        """
        # Channel attention
        x = x * self.channel_attention(x)
        # Spatial attention
        x = x * self.spatial_attention(x)
        return x


class MultiHeadSpatialAttention(nn.Module):
    """
    Multi-head spatial attention for focusing on different regions
    Useful for tracking multiple objects simultaneously
    """
    def __init__(self, channels, num_heads=4):
        super(MultiHeadSpatialAttention, self).__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0
        
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Generate Q, K, V
        q = self.query(x).view(B, self.num_heads, self.head_dim, H * W)
        k = self.key(x).view(B, self.num_heads, self.head_dim, H * W)
        v = self.value(x).view(B, self.num_heads, self.head_dim, H * W)
        
        # Attention scores
        attn = torch.matmul(q.transpose(-2, -1), k)  # (B, num_heads, H*W, H*W)
        attn = F.softmax(attn * (self.head_dim ** -0.5), dim=-1)
        
        # Apply attention
        out = torch.matmul(v, attn.transpose(-2, -1))  # (B, num_heads, head_dim, H*W)
        out = out.view(B, C, H, W)
        out = self.proj(out)
        
        return out + x  # Residual connection


class CrossAttention(nn.Module):
    """
    Cross attention between two feature sets
    Useful for matching detection features with tracking features
    """
    def __init__(self, query_dim, key_dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        
        assert query_dim % num_heads == 0
        
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_dim, query_dim)
        self.v_proj = nn.Linear(key_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, query, key, value):
        """
        Args:
            query: (B, N, query_dim) - e.g., current detections
            key: (B, M, key_dim) - e.g., tracked objects
            value: (B, M, key_dim) - e.g., tracked object features
        Returns:
            (B, N, query_dim) - attended queries
        """
        B, N, _ = query.shape
        M = key.shape[1]
        
        # Project to Q, K, V
        q = self.q_proj(query).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        out = self.out_proj(out)
        
        return out


if __name__ == "__main__":
    print("Testing Attention Modules:\n")
    
    # Test CBAM
    x = torch.randn(2, 256, 40, 40)
    cbam = CBAM(256)
    out = cbam(x)
    print(f"CBAM - Input: {x.shape}, Output: {out.shape}")
    
    # Test MultiHeadSpatialAttention
    mhsa = MultiHeadSpatialAttention(256, num_heads=8)
    out = mhsa(x)
    print(f"MultiHeadSpatialAttention - Input: {x.shape}, Output: {out.shape}")
    
    # Test CrossAttention
    query = torch.randn(2, 10, 256)  # 10 detections
    key = torch.randn(2, 5, 256)     # 5 tracked objects
    value = torch.randn(2, 5, 256)
    cross_attn = CrossAttention(256, 256, num_heads=8)
    out = cross_attn(query, key, value)
    print(f"CrossAttention - Query: {query.shape}, Output: {out.shape}")
    
    print(f"\nTotal CBAM parameters: {sum(p.numel() for p in cbam.parameters()):,}")