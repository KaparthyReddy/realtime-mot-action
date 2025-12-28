import unittest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mot_action_net import MOTActionNet, MOTActionNetLite
from models.backbone.feature_extractor import FeatureExtractor
from models.backbone.temporal_encoder import TemporalEncoder
from models.heads.detection_head import DetectionHead
from models.heads.tracking_head import TrackingHead
from models.heads.action_head import ActionHead


class TestModels(unittest.TestCase):
    """Test model architectures"""
    
    def setUp(self):
        self.batch_size = 2
        self.temporal_window = 8
        self.height = 320
        self.width = 320
        self.num_classes = 5
        self.num_actions = 10
    
    def test_feature_extractor(self):
        """Test feature extractor"""
        model = FeatureExtractor(in_channels=3, base_channels=64)
        x = torch.randn(self.batch_size, 3, self.height, self.width)
        
        features = model(x)
        
        self.assertIn('C3', features)
        self.assertIn('C4', features)
        self.assertIn('C5', features)
        self.assertEqual(features['C3'].shape[0], self.batch_size)
        print("✓ FeatureExtractor test passed")
    
    def test_temporal_encoder(self):
        """Test temporal encoder"""
        model = TemporalEncoder(channels=256, num_layers=3, num_heads=8)
        x = torch.randn(self.batch_size, self.temporal_window, 256)
        
        output = model(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.temporal_window, 256))
        print("✓ TemporalEncoder test passed")
    
    def test_detection_head(self):
        """Test detection head"""
        model = DetectionHead(in_channels=256, num_classes=self.num_classes, num_anchors=3)
        x = torch.randn(self.batch_size, 256, 40, 40)
        
        predictions = model(x)
        
        self.assertIn('bbox', predictions)
        self.assertIn('obj', predictions)
        self.assertIn('cls', predictions)
        print("✓ DetectionHead test passed")
    
    def test_tracking_head(self):
        """Test tracking head"""
        model = TrackingHead(in_channels=256, reid_dim=128)
        x = torch.randn(self.batch_size, 256, 40, 40)
        
        reid_features = model(x)
        
        self.assertEqual(reid_features.shape, (self.batch_size, 128))
        print("✓ TrackingHead test passed")
    
    def test_action_head(self):
        """Test action head"""
        model = ActionHead(in_channels=256, num_classes=self.num_actions, temporal_window=self.temporal_window)
        x = torch.randn(self.batch_size, self.temporal_window, 256)
        
        logits = model(x)
        
        self.assertEqual(logits.shape, (self.batch_size, self.num_actions))
        print("✓ ActionHead test passed")
    
    def test_mot_action_net(self):
        """Test complete MOTActionNet"""
        model = MOTActionNet(
            num_classes=self.num_classes,
            num_actions=self.num_actions,
            temporal_window=self.temporal_window
        )
        
        # Single frame
        x_single = torch.randn(self.batch_size, 3, self.height, self.width)
        out_single = model(x_single, mode='all')
        
        self.assertIn('detection', out_single)
        self.assertIn('tracking', out_single)
        
        # Temporal mode
        x_temporal = torch.randn(self.batch_size, self.temporal_window, 3, self.height, self.width)
        out_temporal = model(x_temporal, mode='all')
        
        self.assertIn('detection', out_temporal)
        self.assertIn('tracking', out_temporal)
        self.assertIn('action', out_temporal)
        
        print("✓ MOTActionNet test passed")
    
    def test_mot_action_net_lite(self):
        """Test MOTActionNetLite"""
        model = MOTActionNetLite(
            num_classes=self.num_classes,
            num_actions=self.num_actions,
            temporal_window=self.temporal_window
        )
        
        x = torch.randn(self.batch_size, self.temporal_window, 3, self.height, self.width)
        output = model(x, mode='all')
        
        self.assertIn('action', output)
        print("✓ MOTActionNetLite test passed")


if __name__ == '__main__':
    unittest.main()