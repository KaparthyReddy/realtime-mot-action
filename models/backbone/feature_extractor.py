import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels, stride=stride)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        self.skip = nn.Sequential()
        if stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels)
            )
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn2(self.conv2(out))
        out += self.skip(residual)
        out = F.relu(out)
        return out


class FeatureExtractor(nn.Module):
    """
    Custom CNN backbone for spatial feature extraction.
    
    Architecture produces multi-scale features:
    - C3: 1/8 resolution (for detection)
    - C4: 1/16 resolution (main features)
    - C5: 1/32 resolution (context)
    
    Args:
        in_channels (int): Number of input channels (default: 3 for RGB)
        base_channels (int): Base number of channels (default: 64)
    """
    def __init__(self, in_channels=3, base_channels=64):
        super(FeatureExtractor, self).__init__()
        
        # Stem: Initial downsampling
        self.stem = nn.Sequential(
            ConvBlock(in_channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Stage 1: 1/4 resolution
        self.stage1 = nn.Sequential(
            ResidualBlock(base_channels),
            ResidualBlock(base_channels)
        )
        
        # Stage 2: 1/8 resolution (C3)
        self.stage2 = nn.Sequential(
            ConvBlock(base_channels, base_channels * 2, stride=2),
            ResidualBlock(base_channels * 2),
            ResidualBlock(base_channels * 2)
        )
        
        # Stage 3: 1/16 resolution (C4)
        self.stage3 = nn.Sequential(
            ConvBlock(base_channels * 2, base_channels * 4, stride=2),
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4)
        )
        
        # Stage 4: 1/32 resolution (C5)
        self.stage4 = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 8, stride=2),
            ResidualBlock(base_channels * 8),
            ResidualBlock(base_channels * 8)
        )
        
        self.out_channels = {
            'C3': base_channels * 2,   # 128
            'C4': base_channels * 4,   # 256
            'C5': base_channels * 8    # 512
        }
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Dictionary of multi-scale features:
            - C3: (B, 128, H/8, W/8)
            - C4: (B, 256, H/16, W/16)
            - C5: (B, 512, H/32, W/32)
        """
        x = self.stem(x)      # (B, 64, H/4, W/4)
        x = self.stage1(x)    # (B, 64, H/4, W/4)
        
        C3 = self.stage2(x)   # (B, 128, H/8, W/8)
        C4 = self.stage3(C3)  # (B, 256, H/16, W/16)
        C5 = self.stage4(C4)  # (B, 512, H/32, W/32)
        
        return {
            'C3': C3,
            'C4': C4,
            'C5': C5
        }


if __name__ == "__main__":
    # Test the feature extractor
    model = FeatureExtractor()
    x = torch.randn(2, 3, 640, 640)
    features = model(x)
    
    print("Feature Extractor Test:")
    for name, feat in features.items():
        print(f"{name}: {feat.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")