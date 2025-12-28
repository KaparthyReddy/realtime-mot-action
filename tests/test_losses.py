import unittest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from losses.detection_loss import DetectionLoss
from losses.tracking_loss import TrackingLoss
from losses.action_loss import ActionLoss
from losses.combined_loss import CombinedLoss


class TestLosses(unittest.TestCase):
    """Test loss functions"""
    
    def test_detection_loss(self):
        """Test detection loss"""
        loss_fn = DetectionLoss()
        
        predictions = {
            'bbox': torch.randn(2, 100, 4),
            'obj': torch.randn(2, 100),
            'cls': torch.randn(2, 100, 5)
        }
        
        targets = {
            'boxes': torch.rand(2, 10, 4),
            'labels': torch.randint(0, 5, (2, 10)),
            'obj_mask': torch.zeros(2, 100, dtype=torch.bool)
        }
        targets['obj_mask'][:, :10] = True
        
        losses = loss_fn(predictions, targets)
        
        self.assertIn('detection_loss', losses)
        self.assertIsInstance(losses['detection_loss'].item(), float)
        print("✓ DetectionLoss test passed")
    
    def test_tracking_loss(self):
        """Test tracking loss"""
        loss_fn = TrackingLoss()
        
        reid_features = torch.randn(2, 50, 128)
        track_ids = torch.randint(0, 20, (2, 50))
        
        losses = loss_fn(reid_features, track_ids)
        
        self.assertIn('tracking_loss', losses)
        print("✓ TrackingLoss test passed")
    
    def test_action_loss(self):
        """Test action loss"""
        loss_fn = ActionLoss(num_classes=10)
        
        predictions = torch.randn(32, 10)
        targets = torch.randint(0, 10, (32,))
        
        losses = loss_fn(predictions, targets)
        
        self.assertIn('action_loss', losses)
        self.assertIn('action_accuracy', losses)
        print("✓ ActionLoss test passed")
    
    def test_combined_loss(self):
        """Test combined loss"""
        loss_fn = CombinedLoss(num_classes=5, num_actions=10)
        
        predictions = {
            'detection': {
                'bbox': torch.randn(2, 50, 4),
                'obj': torch.randn(2, 50),
                'cls': torch.randn(2, 50, 5)
            },
            'tracking': torch.randn(2, 50, 128),
            'action': torch.randn(2, 10)
        }
        
        targets = {
            'boxes': torch.rand(2, 10, 4),
            'labels': torch.randint(0, 5, (2, 10)),
            'obj_mask': torch.zeros(2, 50, dtype=torch.bool),
            'track_ids': torch.randint(0, 20, (2, 50)),
            'actions': torch.randint(0, 10, (2,))
        }
        targets['obj_mask'][:, :10] = True
        
        losses = loss_fn(predictions, targets)
        
        self.assertIn('total_loss', losses)
        self.assertGreaterEqual(losses['total_loss'].item(), 0)
        print("✓ CombinedLoss test passed")


if __name__ == '__main__':
    unittest.main()