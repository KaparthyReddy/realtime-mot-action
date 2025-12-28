import torch
import torch.nn as nn

from .detection_loss import DetectionLoss
from .tracking_loss import TrackingLoss
from .action_loss import ActionLoss


class CombinedLoss(nn.Module):
    """
    Combined loss for multi-task learning
    Balances detection, tracking, and action recognition losses
    
    Args:
        num_classes (int): Number of object classes
        num_actions (int): Number of action classes
        lambda_detection (float): Weight for detection loss
        lambda_tracking (float): Weight for tracking loss
        lambda_action (float): Weight for action loss
        use_uncertainty_weighting (bool): Use learnable uncertainty-based weighting
    """
    def __init__(
        self,
        num_classes=1,
        num_actions=10,
        lambda_detection=1.0,
        lambda_tracking=0.5,
        lambda_action=1.0,
        use_uncertainty_weighting=False
    ):
        super(CombinedLoss, self).__init__()
        
        self.lambda_detection = lambda_detection
        self.lambda_tracking = lambda_tracking
        self.lambda_action = lambda_action
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        # Individual loss modules
        self.detection_loss = DetectionLoss()
        self.tracking_loss = TrackingLoss()
        self.action_loss = ActionLoss(num_classes=num_actions)
        
        # Learnable task weights based on uncertainty
        if use_uncertainty_weighting:
            self.log_vars = nn.Parameter(torch.zeros(3))
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Dictionary with keys ['detection', 'tracking', 'action']
            targets: Dictionary with ground truth data
        
        Returns:
            Dictionary of losses
        """
        losses = {}
        total_loss = 0.0
        
        # 1. Detection Loss
        if 'detection' in predictions and predictions['detection'] is not None:
            det_targets = {
                'boxes': targets.get('boxes'),
                'labels': targets.get('labels'),
                'obj_mask': targets.get('obj_mask')
            }
            
            det_losses = self.detection_loss(predictions['detection'], det_targets)
            
            if self.use_uncertainty_weighting:
                weighted_det_loss = torch.exp(-self.log_vars[0]) * det_losses['detection_loss'] + self.log_vars[0]
            else:
                weighted_det_loss = self.lambda_detection * det_losses['detection_loss']
            
            losses.update(det_losses)
            total_loss += weighted_det_loss
        
        # 2. Tracking Loss
        if 'tracking' in predictions and predictions['tracking'] is not None:
            reid_features = predictions['tracking']
            if reid_features.dim() == 2:
                reid_features = reid_features.unsqueeze(1)
            
            track_ids = targets.get('track_ids')
            if track_ids is not None:
                if track_ids.dim() == 1:
                    track_ids = track_ids.unsqueeze(1)
                
                track_losses = self.tracking_loss(reid_features, track_ids)
                
                if self.use_uncertainty_weighting:
                    weighted_track_loss = torch.exp(-self.log_vars[1]) * track_losses['tracking_loss'] + self.log_vars[1]
                else:
                    weighted_track_loss = self.lambda_tracking * track_losses['tracking_loss']
                
                losses.update(track_losses)
                total_loss += weighted_track_loss
        
        # 3. Action Loss
        if 'action' in predictions and predictions['action'] is not None:
            action_targets = targets.get('actions')
            
            if action_targets is not None:
                action_losses = self.action_loss(predictions['action'], action_targets)
                
                if self.use_uncertainty_weighting:
                    weighted_action_loss = torch.exp(-self.log_vars[2]) * action_losses['action_loss'] + self.log_vars[2]
                else:
                    weighted_action_loss = self.lambda_action * action_losses['action_loss']
                
                losses.update(action_losses)
                total_loss += weighted_action_loss
        
        # Ensure total loss is positive
        losses['total_loss'] = torch.abs(total_loss) if total_loss < 0 else total_loss
        
        # Add task weights for monitoring
        if self.use_uncertainty_weighting:
            losses['weight_detection'] = torch.exp(-self.log_vars[0])
            losses['weight_tracking'] = torch.exp(-self.log_vars[1])
            losses['weight_action'] = torch.exp(-self.log_vars[2])
        
        return losses
    
    def get_task_weights(self):
        """Get current task weights (for logging)"""
        if self.use_uncertainty_weighting:
            weights = torch.exp(-self.log_vars).detach().cpu().numpy()
            return {
                'detection': weights[0],
                'tracking': weights[1],
                'action': weights[2]
            }
        else:
            return {
                'detection': self.lambda_detection,
                'tracking': self.lambda_tracking,
                'action': self.lambda_action
            }