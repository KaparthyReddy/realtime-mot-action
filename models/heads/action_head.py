import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionHead(nn.Module):
    """
    Action Recognition Head for classifying actions performed by detected objects.
    
    Takes temporal features and predicts action classes.
    
    Args:
        in_channels (int): Input feature channels
        num_classes (int): Number of action classes
        temporal_window (int): Number of frames to consider
        hidden_dim (int): Hidden dimension for temporal modeling
    """
    def __init__(self, in_channels=256, num_classes=10, temporal_window=8, hidden_dim=512):
        super(ActionHead, self).__init__()
        
        self.num_classes = num_classes
        self.temporal_window = temporal_window
        self.hidden_dim = hidden_dim
        
        # Spatial feature extraction per frame
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Temporal modeling with LSTM
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Temporal attention
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Action classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_features=False):
        """
        Args:
            x: Temporal features of shape:
               - (B, T, C, H, W) if input is feature maps
               - (B, T, C) if input is already pooled
            return_features: If True, return intermediate features
        
        Returns:
            Action logits (B, num_classes)
        """
        if x.dim() == 5:  # (B, T, C, H, W)
            B, T, C, H, W = x.shape
            
            # Process each frame
            x = x.view(B * T, C, H, W)
            x = self.spatial_conv(x)
            x = self.global_pool(x).view(B, T, C)
        else:  # (B, T, C)
            B, T, C = x.shape
        
        # Temporal modeling with LSTM
        lstm_out, _ = self.lstm(x)  # (B, T, hidden_dim*2)
        
        # Temporal attention
        attn_weights = self.temporal_attention(lstm_out)  # (B, T, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Weighted sum of LSTM outputs
        attended_features = torch.sum(lstm_out * attn_weights, dim=1)  # (B, hidden_dim*2)
        
        # Action classification
        action_logits = self.classifier(attended_features)  # (B, num_classes)
        
        if return_features:
            return action_logits, attended_features
        return action_logits
    
    def predict(self, x, temperature=1.0):
        """
        Predict action classes with temperature scaling
        
        Args:
            x: Input features
            temperature: Temperature for softmax (higher = more uniform)
        
        Returns:
            dict with:
            - 'logits': (B, num_classes)
            - 'probs': (B, num_classes)
            - 'pred_class': (B,)
            - 'confidence': (B,)
        """
        logits = self.forward(x)
        
        # Apply temperature scaling
        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)
        
        confidence, pred_class = torch.max(probs, dim=-1)
        
        return {
            'logits': logits,
            'probs': probs,
            'pred_class': pred_class,
            'confidence': confidence
        }


class SpatialTemporalActionHead(nn.Module):
    """
    Advanced action head with both spatial and temporal attention
    Better for complex actions that involve specific body parts
    """
    def __init__(self, in_channels=256, num_classes=10, temporal_window=8):
        super(SpatialTemporalActionHead, self).__init__()
        
        self.num_classes = num_classes
        self.temporal_window = temporal_window
        
        # Spatial attention module
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, 1),
            nn.Sigmoid()
        )
        
        # 3D convolutions for spatio-temporal modeling
        self.st_conv1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )
        
        self.st_conv2 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels * 2, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_channels * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_channels, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W)
        
        Returns:
            (B, num_classes)
        """
        B, T, C, H, W = x.shape
        
        # Apply spatial attention to each frame
        x_reshaped = x.view(B * T, C, H, W)
        spatial_attn = self.spatial_attention(x_reshaped)
        x_reshaped = x_reshaped * spatial_attn
        x = x_reshaped.view(B, T, C, H, W)
        
        # Rearrange for 3D conv: (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        
        # Spatio-temporal convolutions
        x = self.st_conv1(x)
        x = self.st_conv2(x)
        
        # Flatten and classify
        x = x.view(B, -1)
        logits = self.classifier(x)
        
        return logits


if __name__ == "__main__":
    print("Testing Action Head:\n")
    
    # Create action head
    action_head = ActionHead(in_channels=256, num_classes=10, temporal_window=8)
    
    # Test with feature maps
    x = torch.randn(2, 8, 256, 40, 40)  # 2 videos, 8 frames each
    logits = action_head(x)
    print(f"Action logits (from feature maps): {logits.shape}")
    
    # Test with pooled features
    x_pooled = torch.randn(2, 8, 256)
    logits_pooled = action_head(x_pooled)
    print(f"Action logits (from pooled features): {logits_pooled.shape}")
    
    # Test prediction
    predictions = action_head.predict(x_pooled)
    print(f"\nPredicted classes: {predictions['pred_class']}")
    print(f"Confidence scores: {predictions['confidence']}")
    
    # Test spatial-temporal action head
    st_action_head = SpatialTemporalActionHead(in_channels=256, num_classes=10)
    logits_st = st_action_head(x)
    print(f"\nSpatial-Temporal Action logits: {logits_st.shape}")
    
    print(f"\nAction Head parameters: {sum(p.numel() for p in action_head.parameters()):,}")
    print(f"ST-Action Head parameters: {sum(p.numel() for p in st_action_head.parameters()):,}")