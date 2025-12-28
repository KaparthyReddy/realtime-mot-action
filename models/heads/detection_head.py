import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionHead(nn.Module):
    """
    Detection Head for predicting bounding boxes and class probabilities.
    
    Outputs:
    - Bounding box coordinates (x, y, w, h)
    - Objectness score
    - Class probabilities
    
    Args:
        in_channels (int): Input feature channels
        num_classes (int): Number of object classes (excluding background)
        num_anchors (int): Number of anchor boxes per location
        prior_prob (float): Prior probability for class prediction initialization
    """
    def __init__(self, in_channels=256, num_classes=1, num_anchors=3, prior_prob=0.01):
        super(DetectionHead, self).__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Shared convolutions for feature refinement
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Bounding box regression (4 coords per anchor: x, y, w, h)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, 3, padding=1)
        
        # Objectness prediction (1 score per anchor)
        self.obj_pred = nn.Conv2d(in_channels, num_anchors, 3, padding=1)
        
        # Class prediction (num_classes per anchor)
        self.cls_pred = nn.Conv2d(in_channels, num_anchors * num_classes, 3, padding=1)
        
        # Initialize weights
        self._initialize_weights(prior_prob)
    
    def _initialize_weights(self, prior_prob):
        """Initialize weights with proper bias for stable training"""
        for module in [self.conv1, self.conv2, self.conv3]:
            for layer in module:
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
        
        # Initialize bbox prediction
        nn.init.normal_(self.bbox_pred.weight, std=0.01)
        nn.init.constant_(self.bbox_pred.bias, 0)
        
        # Initialize objectness with slight bias toward no object
        nn.init.normal_(self.obj_pred.weight, std=0.01)
        nn.init.constant_(self.obj_pred.bias, 0)
        
        # Initialize class prediction with prior probability
        nn.init.normal_(self.cls_pred.weight, std=0.01)
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        nn.init.constant_(self.cls_pred.bias, bias_value)
    
    def forward(self, x):
        """
        Args:
            x: Feature map of shape (B, C, H, W)
        
        Returns:
            dict with keys:
            - 'bbox': (B, num_anchors*4, H, W) - bbox predictions
            - 'obj': (B, num_anchors, H, W) - objectness scores
            - 'cls': (B, num_anchors*num_classes, H, W) - class probabilities
        """
        # Feature refinement
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Predictions
        bbox_pred = self.bbox_pred(x)
        obj_pred = self.obj_pred(x)
        cls_pred = self.cls_pred(x)
        
        return {
            'bbox': bbox_pred,
            'obj': obj_pred,
            'cls': cls_pred
        }
    
    def decode_predictions(self, predictions, anchors, conf_threshold=0.5):
        """
        Decode network predictions to actual bounding boxes
        
        Args:
            predictions: Dict from forward()
            anchors: Anchor boxes (num_anchors, 4) in format (cx, cy, w, h)
            conf_threshold: Confidence threshold for filtering
        
        Returns:
            List of detections per image: (x1, y1, x2, y2, objectness, class_id, class_prob)
        """
        bbox_pred = predictions['bbox']  # (B, num_anchors*4, H, W)
        obj_pred = predictions['obj']    # (B, num_anchors, H, W)
        cls_pred = predictions['cls']    # (B, num_anchors*num_classes, H, W)
        
        B, _, H, W = bbox_pred.shape
        device = bbox_pred.device
        
        # Reshape predictions
        bbox_pred = bbox_pred.view(B, self.num_anchors, 4, H, W).permute(0, 1, 3, 4, 2)  # (B, A, H, W, 4)
        obj_pred = obj_pred.view(B, self.num_anchors, H, W).permute(0, 1, 2, 3)  # (B, A, H, W)
        cls_pred = cls_pred.view(B, self.num_anchors, self.num_classes, H, W).permute(0, 1, 3, 4, 2)  # (B, A, H, W, C)
        
        # Apply sigmoid to objectness and class predictions
        obj_scores = torch.sigmoid(obj_pred)
        cls_scores = torch.sigmoid(cls_pred)
        
        detections = []
        for b in range(B):
            batch_detections = []
            
            # Get grid coordinates
            for a in range(self.num_anchors):
                for i in range(H):
                    for j in range(W):
                        obj_score = obj_scores[b, a, i, j].item()
                        
                        if obj_score > conf_threshold:
                            # Decode bbox
                            bbox = bbox_pred[b, a, i, j]  # (4,)
                            
                            # Get class prediction
                            class_scores = cls_scores[b, a, i, j]  # (num_classes,)
                            class_prob, class_id = torch.max(class_scores, dim=0)
                            
                            batch_detections.append({
                                'bbox': bbox.detach().cpu().numpy(),
                                'objectness': obj_score,
                                'class_id': class_id.item(),
                                'class_prob': class_prob.item(),
                                'grid_pos': (i, j),
                                'anchor_id': a
                            })
            
            detections.append(batch_detections)
        
        return detections


class AnchorGenerator:
    """
    Generates anchor boxes for detection
    """
    def __init__(self, sizes=[32, 64, 128], aspect_ratios=[0.5, 1.0, 2.0]):
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.num_anchors = len(sizes) * len(aspect_ratios)
    
    def generate_anchors(self, feature_size, stride):
        """
        Generate anchor boxes for a given feature map
        
        Args:
            feature_size: (H, W) of feature map
            stride: Feature map stride relative to input image
        
        Returns:
            Tensor of shape (H*W*num_anchors, 4) containing (cx, cy, w, h)
        """
        H, W = feature_size
        anchors = []
        
        for i in range(H):
            for j in range(W):
                cx = (j + 0.5) * stride
                cy = (i + 0.5) * stride
                
                for size in self.sizes:
                    for ratio in self.aspect_ratios:
                        w = size * torch.sqrt(torch.tensor(ratio))
                        h = size / torch.sqrt(torch.tensor(ratio))
                        anchors.append([cx, cy, w, h])
        
        return torch.tensor(anchors)


if __name__ == "__main__":
    print("Testing Detection Head:\n")
    
    # Create detection head
    det_head = DetectionHead(in_channels=256, num_classes=5, num_anchors=3)
    
    # Test forward pass
    x = torch.randn(2, 256, 40, 40)
    predictions = det_head(x)
    
    print("Detection Head Outputs:")
    for key, value in predictions.items():
        print(f"  {key}: {value.shape}")
    
    # Test anchor generation
    anchor_gen = AnchorGenerator()
    anchors = anchor_gen.generate_anchors((40, 40), stride=16)
    print(f"\nGenerated anchors: {anchors.shape}")
    
    # Test decoding
    detections = det_head.decode_predictions(predictions, anchors, conf_threshold=0.1)
    print(f"\nNumber of detections per image: {[len(d) for d in detections]}")
    
    print(f"\nTotal parameters: {sum(p.numel() for p in det_head.parameters()):,}")