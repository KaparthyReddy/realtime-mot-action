import torch
import torch.nn as nn
import torch.nn.functional as F


class TrackingHead(nn.Module):
    """
    Tracking Head for generating re-identification (ReID) features.
    
    These features are used to match detections across frames and
    maintain object identities during tracking.
    
    Args:
        in_channels (int): Input feature channels
        reid_dim (int): Dimension of ReID feature vector
        num_layers (int): Number of conv layers before ReID embedding
    """
    def __init__(self, in_channels=256, reid_dim=128, num_layers=2):
        super(TrackingHead, self).__init__()
        
        self.reid_dim = reid_dim
        
        # Feature refinement layers
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            ])
        self.conv_layers = nn.Sequential(*layers)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # ReID embedding layer
        self.reid_embedding = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(in_channels // 2, reid_dim)
        )
        
        # L2 normalization for cosine similarity
        self.normalize = True
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, bboxes=None):
        """
        Args:
            x: Feature map of shape (B, C, H, W)
            bboxes: Optional bounding boxes (B, N, 4) for RoI pooling
                   If provided, extracts features for each bbox
        
        Returns:
            ReID features:
            - If bboxes is None: (B, reid_dim)
            - If bboxes provided: (B, N, reid_dim)
        """
        # Refine features
        x = self.conv_layers(x)
        
        if bboxes is not None:
            # Extract features for each bounding box
            reid_features = self._extract_bbox_features(x, bboxes)
        else:
            # Global features
            x = self.global_pool(x)  # (B, C, 1, 1)
            x = x.view(x.size(0), -1)  # (B, C)
            reid_features = self.reid_embedding(x)  # (B, reid_dim)
        
        # L2 normalize for cosine similarity
        if self.normalize:
            reid_features = F.normalize(reid_features, p=2, dim=-1)
        
        return reid_features
    
    def _extract_bbox_features(self, feature_map, bboxes):
        """
        Extract ReID features for each bounding box using RoI Align
        
        Args:
            feature_map: (B, C, H, W)
            bboxes: (B, N, 4) in format (x1, y1, x2, y2) normalized to [0, 1]
        
        Returns:
            (B, N, reid_dim)
        """
        B, C, H, W = feature_map.shape
        N = bboxes.shape[1]
        
        reid_features = []
        
        for b in range(B):
            batch_features = []
            for n in range(N):
                bbox = bboxes[b, n]  # (4,)
                
                # Convert normalized coords to feature map coords
                x1 = int(bbox[0] * W)
                y1 = int(bbox[1] * H)
                x2 = int(bbox[2] * W)
                y2 = int(bbox[3] * H)
                
                # Clamp to valid range
                x1, x2 = max(0, x1), min(W, x2)
                y1, y2 = max(0, y1), min(H, y2)
                
                # Extract region
                if x2 > x1 and y2 > y1:
                    roi = feature_map[b:b+1, :, y1:y2, x1:x2]  # (1, C, h, w)
                    
                    # Pool to fixed size
                    roi = F.adaptive_avg_pool2d(roi, (7, 7))  # (1, C, 7, 7)
                    roi = self.global_pool(roi)  # (1, C, 1, 1)
                    roi = roi.view(-1)  # (C,)
                    
                    # Get ReID embedding
                    reid_feat = self.reid_embedding(roi.unsqueeze(0))  # (1, reid_dim)
                    batch_features.append(reid_feat)
                else:
                    # Invalid bbox, return zeros
                    batch_features.append(torch.zeros(1, self.reid_dim, device=feature_map.device))
            
            batch_features = torch.cat(batch_features, dim=0)  # (N, reid_dim)
            reid_features.append(batch_features)
        
        reid_features = torch.stack(reid_features, dim=0)  # (B, N, reid_dim)
        
        if self.normalize:
            reid_features = F.normalize(reid_features, p=2, dim=-1)
        
        return reid_features
    
    def compute_similarity(self, features1, features2):
        """
        Compute cosine similarity between two sets of ReID features
        
        Args:
            features1: (B, N, reid_dim) or (N, reid_dim)
            features2: (B, M, reid_dim) or (M, reid_dim)
        
        Returns:
            Similarity matrix (B, N, M) or (N, M)
        """
        # Normalize if not already normalized
        if not self.normalize:
            features1 = F.normalize(features1, p=2, dim=-1)
            features2 = F.normalize(features2, p=2, dim=-1)
        
        # Compute cosine similarity
        if features1.dim() == 3:  # Batched
            similarity = torch.bmm(features1, features2.transpose(1, 2))
        else:  # Unbatched
            similarity = torch.mm(features1, features2.t())
        
        return similarity


class MotionHead(nn.Module):
    """
    Motion Head for predicting object motion (velocity, acceleration)
    Helps improve tracking by predicting where objects will move
    """
    def __init__(self, in_channels=256, motion_dim=4):
        super(MotionHead, self).__init__()
        
        self.motion_dim = motion_dim  # [dx, dy, dw, dh]
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.motion_pred = nn.Linear(in_channels // 2, motion_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize motion prediction to zero (no motion by default)"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize motion prediction bias to zero
        nn.init.constant_(self.motion_pred.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: Feature map (B, C, H, W)
        
        Returns:
            Motion prediction (B, motion_dim) - [dx, dy, dw, dh]
        """
        x = self.conv_layers(x)
        x = self.global_pool(x).view(x.size(0), -1)
        motion = self.motion_pred(x)
        return motion


if __name__ == "__main__":
    print("Testing Tracking Head:\n")
    
    # Create tracking head
    track_head = TrackingHead(in_channels=256, reid_dim=128)
    
    # Test global features
    x = torch.randn(2, 256, 40, 40)
    reid_feats = track_head(x)
    print(f"Global ReID features: {reid_feats.shape}")
    
    # Test with bounding boxes
    bboxes = torch.rand(2, 5, 4)  # 2 images, 5 objects each
    reid_feats_bbox = track_head(x, bboxes)
    print(f"BBox ReID features: {reid_feats_bbox.shape}")
    
    # Test similarity computation
    similarity = track_head.compute_similarity(reid_feats_bbox, reid_feats_bbox)
    print(f"Similarity matrix: {similarity.shape}")
    
    # Test motion head
    motion_head = MotionHead(in_channels=256, motion_dim=4)
    motion_pred = motion_head(x)
    print(f"\nMotion prediction: {motion_pred.shape}")
    
    print(f"\nTracking Head parameters: {sum(p.numel() for p in track_head.parameters()):,}")
    print(f"Motion Head parameters: {sum(p.numel() for p in motion_head.parameters()):,}")