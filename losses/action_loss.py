import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionLoss(nn.Module):
    """
    Loss for action recognition
    
    Supports:
    - Cross-entropy loss for single-label classification
    - Focal loss for handling class imbalance
    - Label smoothing for better generalization
    
    Args:
        num_classes (int): Number of action classes
        label_smoothing (float): Label smoothing factor
        use_focal_loss (bool): Whether to use focal loss
        focal_alpha (float): Focal loss alpha parameter
        focal_gamma (float): Focal loss gamma parameter
    """
    def __init__(
        self,
        num_classes,
        label_smoothing=0.1,
        use_focal_loss=False,
        focal_alpha=0.25,
        focal_gamma=2.0
    ):
        super(ActionLoss, self).__init__()
        
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
    
    def forward(self, predictions, targets, weights=None):
        """
        Args:
            predictions: (B, num_classes) action logits
            targets: (B,) action labels
            weights: Optional (B,) sample weights
        
        Returns:
            Dictionary of losses
        """
        device = predictions.device
        
        if self.use_focal_loss:
            loss = self.focal_loss(predictions, targets)
        else:
            loss = self.cross_entropy_loss(predictions, targets)
        
        # Apply sample weights if provided
        if weights is not None:
            loss = (loss * weights).mean()
        else:
            loss = loss.mean()
        
        # Compute accuracy for monitoring
        with torch.no_grad():
            pred_classes = torch.argmax(predictions, dim=1)
            accuracy = (pred_classes == targets).float().mean()
        
        return {
            'action_loss': loss,
            'action_accuracy': accuracy
        }
    
    def cross_entropy_loss(self, predictions, targets):
        """
        Cross-entropy loss with optional label smoothing
        """
        if self.label_smoothing > 0:
            # Create smooth labels
            n_classes = predictions.size(1)
            one_hot = torch.zeros_like(predictions).scatter(1, targets.unsqueeze(1), 1)
            smooth_labels = one_hot * (1 - self.label_smoothing) + self.label_smoothing / n_classes
            
            log_probs = F.log_softmax(predictions, dim=1)
            loss = -(smooth_labels * log_probs).sum(dim=1)
        else:
            loss = F.cross_entropy(predictions, targets, reduction='none')
        
        return loss
    
    def focal_loss(self, predictions, targets):
        """
        Focal loss for handling class imbalance
        """
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        
        return focal_loss


class TemporalConsistencyLoss(nn.Module):
    """
    Encourages temporal consistency in action predictions
    Penalizes rapid changes in predicted actions across consecutive frames
    """
    def __init__(self, lambda_temporal=0.1):
        super(TemporalConsistencyLoss, self).__init__()
        self.lambda_temporal = lambda_temporal
    
    def forward(self, predictions):
        """
        Args:
            predictions: (B, T, num_classes) action logits over time
        
        Returns:
            Temporal consistency loss
        """
        # Compute softmax probabilities
        probs = F.softmax(predictions, dim=-1)
        
        # Compute differences between consecutive frames
        temporal_diff = torch.abs(probs[:, 1:] - probs[:, :-1])
        
        # Average over time and classes
        loss = temporal_diff.mean() * self.lambda_temporal
        
        return loss


class MultiLabelActionLoss(nn.Module):
    """
    Loss for multi-label action recognition
    (when multiple actions can occur simultaneously)
    """
    def __init__(self, num_classes, pos_weight=None):
        super(MultiLabelActionLoss, self).__init__()
        self.num_classes = num_classes
        
        if pos_weight is not None:
            self.pos_weight = torch.tensor(pos_weight)
        else:
            self.pos_weight = None
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, num_classes) logits
            targets: (B, num_classes) binary labels
        
        Returns:
            Binary cross-entropy loss
        """
        if self.pos_weight is not None:
            loss = F.binary_cross_entropy_with_logits(
                predictions,
                targets.float(),
                pos_weight=self.pos_weight.to(predictions.device)
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                predictions,
                targets.float()
            )
        
        return {
            'action_loss': loss,
            'action_accuracy': self._compute_accuracy(predictions, targets)
        }
    
    def _compute_accuracy(self, predictions, targets):
        """Compute per-sample accuracy for multi-label classification"""
        pred_labels = (torch.sigmoid(predictions) > 0.5).float()
        accuracy = (pred_labels == targets).float().mean()
        return accuracy


class ActionSegmentationLoss(nn.Module):
    """
    Loss for temporal action segmentation
    Predicts action class at each frame in a video
    """
    def __init__(self, num_classes, ignore_index=-1):
        super(ActionSegmentationLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, T, num_classes) per-frame action logits
            targets: (B, T) per-frame action labels
        
        Returns:
            Frame-wise cross-entropy loss
        """
        B, T, C = predictions.shape
        
        # Reshape for cross-entropy
        predictions = predictions.view(B * T, C)
        targets = targets.view(B * T)
        
        # Compute loss (ignore padding frames)
        loss = F.cross_entropy(
            predictions,
            targets,
            ignore_index=self.ignore_index
        )
        
        # Compute frame-wise accuracy
        with torch.no_grad():
            valid_mask = targets != self.ignore_index
            if valid_mask.sum() > 0:
                pred_classes = torch.argmax(predictions[valid_mask], dim=1)
                accuracy = (pred_classes == targets[valid_mask]).float().mean()
            else:
                accuracy = torch.tensor(0.0, device=predictions.device)
        
        return {
            'action_loss': loss,
            'action_accuracy': accuracy
        }


if __name__ == "__main__":
    print("Testing Action Loss:\n")
    
    # Test single-label action loss
    print("1. Single-label Action Loss:")
    action_loss = ActionLoss(num_classes=10, label_smoothing=0.1)
    
    predictions = torch.randn(32, 10)  # 32 samples, 10 classes
    targets = torch.randint(0, 10, (32,))
    
    losses = action_loss(predictions, targets)
    print(f"   Loss: {losses['action_loss'].item():.4f}")
    print(f"   Accuracy: {losses['action_accuracy'].item():.4f}")
    
    # Test focal loss
    print("\n2. Focal Loss:")
    focal_action_loss = ActionLoss(num_classes=10, use_focal_loss=True)
    losses_focal = focal_action_loss(predictions, targets)
    print(f"   Loss: {losses_focal['action_loss'].item():.4f}")
    
    # Test temporal consistency loss
    print("\n3. Temporal Consistency Loss:")
    temporal_loss = TemporalConsistencyLoss(lambda_temporal=0.1)
    temporal_preds = torch.randn(8, 16, 10)  # 8 videos, 16 frames, 10 classes
    tc_loss = temporal_loss(temporal_preds)
    print(f"   Loss: {tc_loss.item():.4f}")
    
    # Test multi-label action loss
    print("\n4. Multi-label Action Loss:")
    multilabel_loss = MultiLabelActionLoss(num_classes=10)
    ml_predictions = torch.randn(32, 10)
    ml_targets = torch.randint(0, 2, (32, 10))
    ml_losses = multilabel_loss(ml_predictions, ml_targets)
    print(f"   Loss: {ml_losses['action_loss'].item():.4f}")
    print(f"   Accuracy: {ml_losses['action_accuracy'].item():.4f}")
    
    # Test action segmentation loss
    print("\n5. Action Segmentation Loss:")
    seg_loss = ActionSegmentationLoss(num_classes=10)
    seg_preds = torch.randn(4, 32, 10)  # 4 videos, 32 frames, 10 classes
    seg_targets = torch.randint(0, 10, (4, 32))
    seg_losses = seg_loss(seg_preds, seg_targets)
    print(f"   Loss: {seg_losses['action_loss'].item():.4f}")
    print(f"   Accuracy: {seg_losses['action_accuracy'].item():.4f}")