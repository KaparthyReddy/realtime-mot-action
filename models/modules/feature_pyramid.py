import torch
import torch.nn as nn
import torch.nn.functional as F


class FPNLayer(nn.Module):
    """Single FPN layer for lateral connection and top-down pathway"""
    def __init__(self, in_channels, out_channels):
        super(FPNLayer, self).__init__()
        self.lateral = nn.Conv2d(in_channels, out_channels, 1)
        self.output = nn.Conv2d(out_channels, out_channels, 3, padding=1)
    
    def forward(self, x, top_down=None):
        """
        Args:
            x: Current level features
            top_down: Upsampled features from higher level (optional)
        """
        lateral = self.lateral(x)
        
        if top_down is not None:
            # Upsample top-down to match lateral size
            top_down = F.interpolate(top_down, size=lateral.shape[-2:], 
                                    mode='nearest')
            lateral = lateral + top_down
        
        output = self.output(lateral)
        return output


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network (FPN)
    
    Creates a multi-scale feature pyramid from backbone features.
    Useful for detecting objects at different scales.
    
    Reference: "Feature Pyramid Networks for Object Detection" (Lin et al., CVPR 2017)
    
    Args:
        in_channels_list (list): List of input channels for each level [C3, C4, C5]
        out_channels (int): Output channels for all pyramid levels
        extra_levels (int): Number of additional pyramid levels
    """
    def __init__(self, in_channels_list=[128, 256, 512], out_channels=256, extra_levels=1):
        super(FeaturePyramidNetwork, self).__init__()
        
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.extra_levels = extra_levels
        
        # Lateral and output convolutions for each input level
        self.fpn_layers = nn.ModuleList([
            FPNLayer(in_ch, out_channels) for in_ch in in_channels_list
        ])
        
        # Extra pyramid levels (for detecting larger objects)
        self.extra_layers = nn.ModuleList()
        for i in range(extra_levels):
            if i == 0:
                in_ch = in_channels_list[-1]
            else:
                in_ch = out_channels
            self.extra_layers.append(
                nn.Conv2d(in_ch, out_channels, 3, stride=2, padding=1)
            )
    
    def forward(self, features):
        """
        Args:
            features: Dictionary with keys ['C3', 'C4', 'C5']
                     containing features of shape (B, C_i, H_i, W_i)
        
        Returns:
            List of pyramid features [P3, P4, P5, P6, ...]
            Each with shape (B, out_channels, H_i, W_i)
        """
        # Extract input features
        c3, c4, c5 = features['C3'], features['C4'], features['C5']
        inputs = [c3, c4, c5]
        
        # Build top-down pathway
        pyramid_features = []
        
        # Start from the top (coarsest resolution)
        top_down = None
        for i in range(len(inputs) - 1, -1, -1):
            top_down = self.fpn_layers[i](inputs[i], top_down)
            pyramid_features.insert(0, top_down)
        
        # Add extra pyramid levels
        last_feature = inputs[-1]  # Start from C5
        for extra_layer in self.extra_layers:
            last_feature = extra_layer(last_feature)
            pyramid_features.append(last_feature)
        
        return pyramid_features


class BiFPN(nn.Module):
    """
    Bidirectional Feature Pyramid Network (BiFPN)
    
    Enhanced FPN with bidirectional cross-scale connections and weighted fusion.
    More powerful than standard FPN but also more computationally expensive.
    
    Reference: "EfficientDet: Scalable and Efficient Object Detection" (Tan et al., CVPR 2020)
    """
    def __init__(self, in_channels_list=[128, 256, 512], out_channels=256, num_layers=2):
        super(BiFPN, self).__init__()
        
        self.num_layers = num_layers
        
        # Input convolutions to standardize channels
        self.input_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        
        # BiFPN layers
        self.bifpn_layers = nn.ModuleList([
            BiFPNLayer(out_channels, len(in_channels_list)) 
            for _ in range(num_layers)
        ])
    
    def forward(self, features):
        """
        Args:
            features: Dictionary with keys ['C3', 'C4', 'C5']
        Returns:
            List of BiFPN features
        """
        # Standardize input channels
        c3, c4, c5 = features['C3'], features['C4'], features['C5']
        inputs = [c3, c4, c5]
        
        feats = [conv(x) for conv, x in zip(self.input_convs, inputs)]
        
        # Apply BiFPN layers
        for bifpn_layer in self.bifpn_layers:
            feats = bifpn_layer(feats)
        
        return feats


class BiFPNLayer(nn.Module):
    """Single BiFPN layer with top-down and bottom-up pathways"""
    def __init__(self, channels, num_levels):
        super(BiFPNLayer, self).__init__()
        self.num_levels = num_levels
        
        # Top-down pathway
        self.td_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for _ in range(num_levels - 1)
        ])
        
        # Bottom-up pathway
        self.bu_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for _ in range(num_levels - 1)
        ])
        
        # Learnable weights for fusion
        self.td_weights = nn.Parameter(torch.ones(num_levels - 1, 2))
        self.bu_weights = nn.Parameter(torch.ones(num_levels - 1, 3))
    
    def forward(self, features):
        """
        Args:
            features: List of features at different scales
        Returns:
            List of fused features
        """
        # Top-down pathway
        td_features = [features[-1]]
        for i in range(len(features) - 2, -1, -1):
            up = F.interpolate(td_features[0], size=features[i].shape[-2:], mode='nearest')
            w = F.relu(self.td_weights[i])
            w = w / (w.sum() + 1e-4)
            fused = w[0] * features[i] + w[1] * up
            fused = self.td_convs[i](fused)
            td_features.insert(0, fused)
        
        # Bottom-up pathway
        bu_features = [td_features[0]]
        for i in range(1, len(features)):
            down = F.max_pool2d(bu_features[-1], kernel_size=2)
            w = F.relu(self.bu_weights[i-1])
            w = w / (w.sum() + 1e-4)
            fused = w[0] * features[i] + w[1] * td_features[i] + w[2] * down
            fused = self.bu_convs[i-1](fused)
            bu_features.append(fused)
        
        return bu_features


if __name__ == "__main__":
    print("Testing Feature Pyramid Modules:\n")
    
    # Simulate backbone features
    features = {
        'C3': torch.randn(2, 128, 80, 80),
        'C4': torch.randn(2, 256, 40, 40),
        'C5': torch.randn(2, 512, 20, 20)
    }
    
    # Test FPN
    fpn = FeaturePyramidNetwork(in_channels_list=[128, 256, 512], 
                                out_channels=256, extra_levels=2)
    pyramid_feats = fpn(features)
    print("FPN Output:")
    for i, feat in enumerate(pyramid_feats):
        print(f"  P{i+3}: {feat.shape}")
    
    # Test BiFPN
    bifpn = BiFPN(in_channels_list=[128, 256, 512], out_channels=256, num_layers=2)
    bifpn_feats = bifpn(features)
    print("\nBiFPN Output:")
    for i, feat in enumerate(bifpn_feats):
        print(f"  Level {i}: {feat.shape}")
    
    print(f"\nFPN parameters: {sum(p.numel() for p in fpn.parameters()):,}")
    print(f"BiFPN parameters: {sum(p.numel() for p in bifpn.parameters()):,}")