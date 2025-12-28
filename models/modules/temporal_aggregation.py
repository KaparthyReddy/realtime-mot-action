import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAggregation(nn.Module):
    """
    Aggregates features across temporal dimension
    
    Supports multiple aggregation strategies:
    - 'max': Max pooling over time
    - 'avg': Average pooling over time
    - 'attention': Learned attention weights
    - 'conv': 3D convolution
    
    Args:
        channels (int): Number of feature channels
        num_frames (int): Number of frames to aggregate
        mode (str): Aggregation mode
    """
    def __init__(self, channels, num_frames=8, mode='attention'):
        super(TemporalAggregation, self).__init__()
        self.channels = channels
        self.num_frames = num_frames
        self.mode = mode
        
        if mode == 'attention':
            self.attention = nn.Sequential(
                nn.Conv2d(channels, channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 4, 1, 1),
                nn.Sigmoid()
            )
        elif mode == 'conv':
            # 3D convolution over temporal dimension
            self.conv3d = nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), 
                                   padding=(1, 0, 0))
            self.bn = nn.BatchNorm3d(channels)
            self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) or (B*T, C, H, W)
        Returns:
            (B, C, H, W) - aggregated features
        """
        if x.dim() == 4:
            # Reshape from (B*T, C, H, W) to (B, T, C, H, W)
            BT, C, H, W = x.shape
            B = BT // self.num_frames
            x = x.view(B, self.num_frames, C, H, W)
        else:
            B, T, C, H, W = x.shape
        
        if self.mode == 'max':
            out, _ = torch.max(x, dim=1)
        
        elif self.mode == 'avg':
            out = torch.mean(x, dim=1)
        
        elif self.mode == 'attention':
            # Compute attention weights for each frame
            attn_weights = []
            for t in range(T):
                attn = self.attention(x[:, t])  # (B, 1, H, W)
                attn_weights.append(attn)
            
            attn_weights = torch.stack(attn_weights, dim=1)  # (B, T, 1, H, W)
            attn_weights = F.softmax(attn_weights, dim=1)
            
            # Weighted sum
            out = (x * attn_weights).sum(dim=1)
        
        elif self.mode == 'conv':
            # 3D convolution
            x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
            x = self.relu(self.bn(self.conv3d(x)))
            out = torch.mean(x, dim=2)  # Average over temporal dimension
        
        else:
            raise ValueError(f"Unknown aggregation mode: {self.mode}")
        
        return out


class AdaptiveTemporalPooling(nn.Module):
    """
    Adaptive temporal pooling that can handle variable-length sequences
    """
    def __init__(self, channels, output_frames=1):
        super(AdaptiveTemporalPooling, self).__init__()
        self.output_frames = output_frames
        self.pool = nn.AdaptiveAvgPool3d((output_frames, None, None))
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            (B, output_frames, C, H, W)
        """
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        x = self.pool(x)
        x = x.permute(0, 2, 1, 3, 4)  # (B, output_frames, C, H, W)
        
        if self.output_frames == 1:
            x = x.squeeze(1)  # (B, C, H, W)
        
        return x


class TemporalShift(nn.Module):
    """
    Temporal Shift Module (TSM) for efficient temporal modeling
    Shifts channels along temporal dimension for free temporal reasoning
    
    Reference: "TSM: Temporal Shift Module for Efficient Video Understanding"
    """
    def __init__(self, channels, num_frames, shift_div=8):
        super(TemporalShift, self).__init__()
        self.channels = channels
        self.num_frames = num_frames
        self.shift_div = shift_div
        self.fold = channels // shift_div
    
    def forward(self, x):
        """
        Args:
            x: (B*T, C, H, W)
        Returns:
            (B*T, C, H, W) with temporal shifts applied
        """
        BT, C, H, W = x.shape
        B = BT // self.num_frames
        T = self.num_frames
        
        x = x.view(B, T, C, H, W)
        
        # Split into 3 parts: backward shift, forward shift, no shift
        out = torch.zeros_like(x)
        
        # Backward shift (shift left along time)
        out[:, :-1, :self.fold] = x[:, 1:, :self.fold]
        
        # Forward shift (shift right along time)
        out[:, 1:, self.fold:2*self.fold] = x[:, :-1, self.fold:2*self.fold]
        
        # No shift
        out[:, :, 2*self.fold:] = x[:, :, 2*self.fold:]
        
        return out.view(BT, C, H, W)


if __name__ == "__main__":
    print("Testing Temporal Aggregation Modules:\n")
    
    # Test TemporalAggregation with different modes
    B, T, C, H, W = 2, 8, 256, 40, 40
    x = torch.randn(B, T, C, H, W)
    
    for mode in ['max', 'avg', 'attention', 'conv']:
        agg = TemporalAggregation(C, num_frames=T, mode=mode)
        out = agg(x)
        print(f"TemporalAggregation ({mode}) - Input: {x.shape}, Output: {out.shape}")
    
    # Test AdaptiveTemporalPooling
    adaptive_pool = AdaptiveTemporalPooling(C, output_frames=1)
    out = adaptive_pool(x)
    print(f"\nAdaptiveTemporalPooling - Input: {x.shape}, Output: {out.shape}")
    
    # Test TemporalShift
    x_flat = torch.randn(B * T, C, H, W)
    tsm = TemporalShift(C, num_frames=T)
    out = tsm(x_flat)
    print(f"TemporalShift - Input: {x_flat.shape}, Output: {out.shape}")