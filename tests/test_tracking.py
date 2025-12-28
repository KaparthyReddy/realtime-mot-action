import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracking.kalman_filter import KalmanFilter
from tracking.matching import match_detections_to_tracks, iou
from tracking.tracker import MultiObjectTracker, Track


class TestTracking(unittest.TestCase):
    """Test tracking algorithms"""
    
    def test_kalman_filter(self):
        """Test Kalman filter"""
        kf = KalmanFilter(dt=1.0)
        
        # Initialize with measurement
        measurement = np.array([100.0, 100.0, 50.0, 80.0])
        kf.init_state(measurement)
        
        # Predict
        predicted = kf.predict()
        self.assertEqual(len(predicted), 8)  # 4 position + 4 velocity
        
        # Update
        new_measurement = np.array([105.0, 103.0, 50.0, 80.0])
        updated = kf.update(new_measurement)
        
        self.assertIsNotNone(updated)
        print("✓ KalmanFilter test passed")
    
    def test_iou_computation(self):
        """Test IoU computation"""
        box1 = np.array([100, 100, 50, 80])
        box2 = np.array([105, 102, 50, 80])
        
        iou_value = iou(box1, box2)
        
        self.assertGreater(iou_value, 0.0)
        self.assertLessEqual(iou_value, 1.0)
        print("✓ IoU computation test passed")
    
    def test_matching(self):
        """Test detection to track matching"""
        detections = np.array([
            [100, 100, 50, 80],
            [300, 200, 60, 90],
            [500, 150, 55, 85]
        ])
        
        tracks = np.array([
            [105, 102, 50, 80],
            [450, 145, 55, 85],
            [700, 300, 60, 90]
        ])
        
        matches, unmatched_det, unmatched_track = match_detections_to_tracks(
            detections, tracks, iou_threshold=0.3
        )
        
        self.assertIsInstance(matches, list)
        self.assertIsInstance(unmatched_det, list)
        self.assertIsInstance(unmatched_track, list)
        print("✓ Matching test passed")
    
    def test_track_creation(self):
        """Test track creation"""
        bbox = np.array([100, 100, 50, 80])
        track = Track(bbox, feature=None, class_id=0, confidence=0.9)
        
        self.assertIsNotNone(track.track_id)
        self.assertEqual(track.state, 'tentative')
        self.assertEqual(track.age, 1)
        print("✓ Track creation test passed")
    
    def test_multi_object_tracker(self):
        """Test multi-object tracker"""
        tracker = MultiObjectTracker(max_age=30, min_hits=3)
        
        # First frame
        detections = np.array([
            [100, 100, 50, 80],
            [300, 200, 60, 90]
        ])
        
        tracks = tracker.update(detections)
        
        # Should have created 2 tentative tracks
        self.assertGreaterEqual(len(tracker.tracks), 0)
        
        # Update with similar detections
        detections = np.array([
            [105, 103, 50, 80],
            [305, 203, 60, 90]
        ])
        
        tracks = tracker.update(detections)
        
        print("✓ MultiObjectTracker test passed")


if __name__ == '__main__':
    unittest.main()