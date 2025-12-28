import torch
import torch.nn as nn
import torch.nn.functional as F

# Use absolute imports from the project root
from models.backbone.feature_extractor import FeatureExtractor
from models.backbone.temporal_encoder import TemporalEncoder
from models.modules.feature_pyramid import FeaturePyramidNetwork
from models.modules.temporal_aggregation import TemporalAggregation
from models.modules.attention import CBAM
from models.heads.detection_head import DetectionHead
from models.heads.tracking_head import TrackingHead
from models.heads.action_head import ActionHead

class MOTActionNet(nn.Module):
    def __init__(
        self,
        num_classes=1,
        num_actions=10,
        backbone_channels=64,
        fpn_channels=256,
        reid_dim=128,
        temporal_window=8,
        use_attention=True
    ):
        super(MOTActionNet, self).__init__()
        
        self.num_classes = num_classes
        self.num_actions = num_actions
        self.fpn_channels = fpn_channels
        self.reid_dim = reid_dim
        self.temporal_window = temporal_window
        self.use_attention = use_attention
        
        # 1. Spatial Feature Extraction
        self.backbone = FeatureExtractor(
            in_channels=3,
            base_channels=backbone_channels
        )
        
        # 2. Feature Pyramid Network
        backbone_out_channels = self.backbone.out_channels
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[
                backbone_out_channels['C3'],
                backbone_out_channels['C4'],
                backbone_out_channels['C5']
            ],
            out_channels=fpn_channels,
            extra_levels=1
        )
        
        # 3. Attention modules
        if use_attention:
            self.attention_modules = nn.ModuleList([
                CBAM(fpn_channels) for _ in range(4)
            ])
        
        # 4. Temporal Encoder (Expects fpn_channels)
        self.temporal_encoder = TemporalEncoder(
            channels=fpn_channels,
            num_layers=3,
            num_heads=8,
            max_seq_len=temporal_window
        )
        
        # 5. Temporal Aggregation
        self.temporal_aggregation = TemporalAggregation(
            channels=fpn_channels,
            num_frames=temporal_window,
            mode='attention'
        )
        
        # 6. Task-specific Heads
        self.detection_head = DetectionHead(
            in_channels=fpn_channels,
            num_classes=num_classes,
            num_anchors=3
        )
        
        self.tracking_head = TrackingHead(
            in_channels=fpn_channels,
            reid_dim=reid_dim,
            num_layers=2
        )
        
        self.action_head = ActionHead(
            in_channels=fpn_channels,
            num_classes=num_actions,
            temporal_window=temporal_window,
            hidden_dim=512
        )

        # 7. Projection layer to bridge ReID features to Temporal Encoder
        # This fixes the dimension mismatch (128 -> 256)
        self.reid_to_temporal = nn.Linear(reid_dim, fpn_channels)
        
        print(f"MOTActionNet initialized:")
        print(f"  - Object classes: {num_classes}")
        print(f"  - Action classes: {num_actions}")
        print(f"  - Temporal window: {temporal_window}")
        print(f"  - ReID dimension: {reid_dim}")
        print(f"  - Using attention: {use_attention}")
    
    def forward(self, x, mode='all'):
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            temporal_mode = True
            x_flat = x.view(B * T, C, H, W)
        else:
            B, C, H, W = x.shape
            T = 1
            temporal_mode = False
            x_flat = x
        
        backbone_features = self.backbone(x_flat)
        pyramid_features = self.fpn(backbone_features)
        
        if self.use_attention:
            pyramid_features = [
                attn(feat) for attn, feat in zip(self.attention_modules, pyramid_features)
            ]
        
        main_features = pyramid_features[1]
        outputs = {}
        
        # Detection
        if mode in ['all', 'detection']:
            detection_out = self.detection_head(main_features)
            if temporal_mode:
                for key in detection_out:
                    shape = detection_out[key].shape
                    detection_out[key] = detection_out[key].view(B, T, *shape[1:])
            outputs['detection'] = detection_out
        
        # Tracking
        if mode in ['all', 'tracking']:
            tracking_features = self.tracking_head(main_features)
            if temporal_mode:
                tracking_features = tracking_features.view(B, T, -1)
            outputs['tracking'] = tracking_features
        
        # Action Recognition
        if mode in ['all', 'action']:
            if temporal_mode:
                _, C, H, W = main_features.shape
                temporal_feats = main_features.view(B, T, C, H, W)
                temporal_feats_pooled = F.adaptive_avg_pool2d(
                    temporal_feats.view(B * T, C, H, W), 1
                ).view(B, T, C)
                
                encoded_temporal = self.temporal_encoder(temporal_feats_pooled)
                action_logits = self.action_head(encoded_temporal)
                outputs['action'] = action_logits
            else:
                outputs['action'] = None
        
        return outputs
    
    def extract_temporal_features(self, x):
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        backbone_features = self.backbone(x_flat)
        pyramid_features = self.fpn(backbone_features)
        main_features = pyramid_features[1]
        
        # Extract ReID features (Size: reid_dim)
        reid_features = self.tracking_head(main_features)
        reid_features = reid_features.view(B, T, -1)
        
        # Project ReID features to match TemporalEncoder's expected fpn_channels
        # Fixes: RuntimeError (size 128 must match size 256)
        projected_features = self.reid_to_temporal(reid_features)
        
        # Encode temporal dependencies
        temporal_features = self.temporal_encoder(projected_features)
        
        return temporal_features

    def compute_losses(self, predictions, targets):
        losses = {}
        if 'detection' in predictions and predictions['detection'] is not None:
            losses['detection_loss'] = torch.tensor(0.0, device=predictions['detection']['bbox'].device)
        if 'tracking' in predictions and predictions['tracking'] is not None:
            losses['tracking_loss'] = torch.tensor(0.0, device=predictions['tracking'].device)
        if 'action' in predictions and predictions['action'] is not None:
            losses['action_loss'] = torch.tensor(0.0, device=predictions['action'].device)
        
        losses['total_loss'] = sum(losses.values())
        return losses

class MOTActionNetLite(MOTActionNet):
    def __init__(self, num_classes=1, num_actions=10, temporal_window=8):
        super().__init__(
            num_classes=num_classes,
            num_actions=num_actions,
            backbone_channels=32,
            fpn_channels=128,
            reid_dim=64,
            temporal_window=temporal_window,
            use_attention=False
        )

if __name__ == "__main__":
    print("Testing MOTActionNet:\n")
    model = MOTActionNet(num_classes=5, num_actions=10, temporal_window=8)
    
    print("\n" + "="*60)
    print("\n1. Testing single frame mode:")
    x_single = torch.randn(2, 3, 640, 640)
    out_single = model(x_single, mode='all')
    print(f"  Tracking features: {out_single['tracking'].shape}")
    
    print("\n2. Testing temporal mode:")
    x_temporal = torch.randn(2, 8, 3, 640, 640)
    out_temporal = model(x_temporal, mode='all')
    print(f"  Action logits: {out_temporal['action'].shape}")
    
    print("\n4. Testing temporal feature extraction (Fix verification):")
    temp_feats = model.extract_temporal_features(x_temporal)
    print(f"  Temporal features: {temp_feats.shape}")
    
    print("\n" + "="*60)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Statistics:")
    print(f"  Total parameters: {total_params:,}")