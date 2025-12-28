import torch
import torch.nn as nn

from .detection_loss import DetectionLoss
from .tracking_loss import TrackingLoss
from .action_loss import ActionLoss


class CombinedLoss(nn.Module):
    """
    Combined loss for multi-task learning with stable uncertainty weighting.
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
        
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        # Individual loss modules
        self.detection_loss = DetectionLoss()
        self.tracking_loss = TrackingLoss()
        self.action_loss = ActionLoss(num_classes=num_actions)
        
        if use_uncertainty_weighting:
            # Initialize log_vars at 0.0 so initial weights are 1.0 (exp(-0))
            # Parameters: [Detection, Tracking, Action]
            self.log_vars = nn.Parameter(torch.zeros(3))
        else:
            # Fixed weights
            self.register_buffer('lambdas', torch.tensor([
                lambda_detection, 
                lambda_tracking, 
                lambda_action
            ]))
    
    def forward(self, predictions, targets):
        losses = {}
        task_losses = [] # To store [det, track, action] for weighting
        
        # 1. Detection Loss
        det_targets = {
            'boxes': targets.get('boxes'),
            'labels': targets.get('labels'),
            'obj_mask': targets.get('obj_mask')
        }
        det_results = self.detection_loss(predictions['detection'], det_targets)
        losses.update(det_results)
        task_losses.append(det_results['detection_loss'])
        
        # 2. Tracking Loss
        reid_features = predictions['tracking']
        if reid_features.dim() == 2:
            reid_features = reid_features.unsqueeze(1)
        
        track_ids = targets.get('track_ids')
        if track_ids is not None:
            if track_ids.dim() == 1:
                track_ids = track_ids.unsqueeze(1)
            track_results = self.tracking_loss(reid_features, track_ids)
            losses.update(track_results)
            task_losses.append(track_results['tracking_loss'])
        else:
            task_losses.append(torch.tensor(0.0, device=reid_features.device))
        
        # 3. Action Loss
        action_targets = targets.get('actions')
        action_results = self.action_loss(predictions['action'], action_targets)
        losses.update(action_results)
        task_losses.append(action_results['action_loss'])
        
        # Combine losses
        if self.use_uncertainty_weighting:
            total_loss = 0.0
            for i in range(3):
                # Standard formula: exp(-s) * L + s
                # We clamp log_vars to prevent log(sigma) from becoming too negative
                s = self.log_vars[i]
                total_loss += torch.exp(-s) * task_losses[i] + s
            
            # Monitoring weights
            losses['weight_detection'] = torch.exp(-self.log_vars[0])
            losses['weight_tracking'] = torch.exp(-self.log_vars[1])
            losses['weight_action'] = torch.exp(-self.log_vars[2])
        else:
            total_loss = (self.lambdas[0] * task_losses[0] + 
                          self.lambdas[1] * task_losses[1] + 
                          self.lambdas[2] * task_losses[2])
        
        losses['total_loss'] = total_loss
        return losses