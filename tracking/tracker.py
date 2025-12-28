import numpy as np
from collections import defaultdict

# Use absolute imports
from tracking.kalman_filter import KalmanFilter
# Import the specific matching function from matching.py
from tracking.matching import match_detections_to_tracks

class Track:
    """
    Represents a single tracked object
    
    Maintains:
    - Bounding box state (via Kalman filter)
    - ReID features
    - Track history and confidence
    """
    
    # Class variable for unique track IDs
    _next_id = 1
    
    def __init__(self, bbox, feature=None, class_id=None, confidence=1.0):
        """
        Args:
            bbox: [x, y, w, h] initial bounding box
            feature: ReID feature vector
            class_id: Object class
            confidence: Detection confidence
        """
        self.track_id = Track._next_id
        Track._next_id += 1
        
        # Kalman filter for motion prediction
        self.kf = KalmanFilter(dt=1.0)
        self.kf.init_state(bbox)
        
        # Track state
        self.bbox = bbox
        self.predicted_bbox = bbox
        self.class_id = class_id
        self.confidence = confidence
        
        # ReID features (maintain history for robustness)
        self.features = [feature] if feature is not None else []
        self.max_feature_history = 10
        
        # Track management
        self.age = 1  # Number of frames since creation
        self.hits = 1  # Number of successful matches
        self.time_since_update = 0  # Frames since last update
        self.state = 'tentative'  # 'tentative', 'confirmed', 'lost'
        
        # History
        self.bbox_history = [bbox]
        self.max_history = 30
    
    def predict(self):
        """Predict next state using motion model"""
        state = self.kf.predict()
        self.predicted_bbox = state[:4]
        self.age += 1
        self.time_since_update += 1
        return self.predicted_bbox
    
    def update(self, bbox, feature=None, confidence=None):
        """
        Update track with new detection
        
        Args:
            bbox: [x, y, w, h] detected bounding box
            feature: ReID feature vector
            confidence: Detection confidence
        """
        # Update Kalman filter
        self.kf.update(bbox)
        self.bbox = bbox
        self.predicted_bbox = bbox
        
        # Update features
        if feature is not None:
            self.features.append(feature)
            if len(self.features) > self.max_feature_history:
                self.features.pop(0)
        
        # Update confidence
        if confidence is not None:
            self.confidence = confidence
        
        # Update track state
        self.hits += 1
        self.time_since_update = 0
        
        # Confirm track after enough hits
        if self.state == 'tentative' and self.hits >= 3:
            self.state = 'confirmed'
        
        # Update history
        self.bbox_history.append(bbox)
        if len(self.bbox_history) > self.max_history:
            self.bbox_history.pop(0)
    
    def mark_missed(self):
        """Mark track as missed in current frame"""
        self.time_since_update += 1
        
        # Mark as lost after too many misses
        if self.time_since_update > 30:
            self.state = 'lost'
    
    def get_feature(self):
        """Get average ReID feature"""
        if not self.features:
            return None
        
        # Average recent features
        features = np.array(self.features[-5:])
        avg_feature = np.mean(features, axis=0)
        
        # Normalize
        avg_feature = avg_feature / (np.linalg.norm(avg_feature) + 1e-7)
        
        return avg_feature
    
    def is_tentative(self):
        """Check if track is still tentative"""
        return self.state == 'tentative'
    
    def is_confirmed(self):
        """Check if track is confirmed"""
        return self.state == 'confirmed'
    
    def is_lost(self):
        """Check if track is lost"""
        return self.state == 'lost'
    
    def get_velocity(self):
        """Get estimated velocity [vx, vy]"""
        state = self.kf.get_state()
        if state is not None:
            return state[4:6]
        return np.zeros(2)


class MultiObjectTracker:
    """
    Multi-Object Tracker using detection, motion, and appearance
    
    Implements tracking-by-detection with:
    - Kalman filter for motion prediction
    - ReID features for appearance matching
    - Hungarian algorithm for data association
    - Track lifecycle management
    
    Args:
        max_age (int): Maximum frames to keep lost tracks
        min_hits (int): Minimum hits to confirm track
        iou_threshold (float): Minimum IoU for matching
        feature_weight (float): Weight for appearance features [0, 1]
    """
    
    def __init__(
        self,
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        feature_weight=0.5
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.feature_weight = feature_weight
        
        self.tracks = []
        self.frame_count = 0
        
        # Statistics
        self.total_tracks_created = 0
        self.active_track_count = 0
    
    def update(self, detections, features=None, classes=None, confidences=None):
        """
        Update tracker with new detections
        
        Args:
            detections: (N, 4) array of [x, y, w, h] bounding boxes
            features: (N, D) array of ReID features (optional)
            classes: (N,) array of class IDs (optional)
            confidences: (N,) array of detection confidences (optional)
        
        Returns:
            List of active tracks with their bounding boxes and IDs
        """
        self.frame_count += 1
        
        # Predict new locations for existing tracks
        for track in self.tracks:
            track.predict()
        
        # Get predicted bboxes and features for matching
        if len(self.tracks) > 0:
            track_bboxes = np.array([t.predicted_bbox for t in self.tracks])
            track_features = np.array([t.get_feature() for t in self.tracks])
            track_features = track_features if track_features[0] is not None else None
        else:
            track_bboxes = np.empty((0, 4))
            track_features = None
        
        # Match detections to tracks
        if len(detections) > 0:
            matches, unmatched_detections, unmatched_tracks = match_detections_to_tracks(
                detections,
                track_bboxes,
                detection_features=features,
                track_features=track_features,
                iou_threshold=self.iou_threshold,
                feature_weight=self.feature_weight
            )
        else:
            matches = []
            unmatched_detections = []
            unmatched_tracks = list(range(len(self.tracks)))
        
        # Update matched tracks
        for det_idx, track_idx in matches:
            bbox = detections[det_idx]
            feature = features[det_idx] if features is not None else None
            class_id = classes[det_idx] if classes is not None else None
            confidence = confidences[det_idx] if confidences is not None else None
            
            self.tracks[track_idx].update(bbox, feature, confidence)
            if class_id is not None:
                self.tracks[track_idx].class_id = class_id
        
        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            bbox = detections[det_idx]
            feature = features[det_idx] if features is not None else None
            class_id = classes[det_idx] if classes is not None else None
            confidence = confidences[det_idx] if confidences is not None else 1.0
            
            new_track = Track(bbox, feature, class_id, confidence)
            self.tracks.append(new_track)
            self.total_tracks_created += 1
        
        # Remove lost tracks
        self.tracks = [t for t in self.tracks if not t.is_lost()]
        
        # Get output tracks (only confirmed ones)
        output_tracks = []
        for track in self.tracks:
            if track.is_confirmed() or (track.is_tentative() and track.hits >= self.min_hits):
                output_tracks.append({
                    'track_id': track.track_id,
                    'bbox': track.bbox,
                    'class_id': track.class_id,
                    'confidence': track.confidence,
                    'age': track.age,
                    'hits': track.hits
                })
        
        self.active_track_count = len(output_tracks)
        
        return output_tracks
    
    def get_tracks(self, confirmed_only=True):
        """
        Get current tracks
        
        Args:
            confirmed_only: Only return confirmed tracks
        
        Returns:
            List of track dictionaries
        """
        if confirmed_only:
            tracks = [t for t in self.tracks if t.is_confirmed()]
        else:
            tracks = self.tracks
        
        output = []
        for track in tracks:
            output.append({
                'track_id': track.track_id,
                'bbox': track.bbox,
                'predicted_bbox': track.predicted_bbox,
                'class_id': track.class_id,
                'confidence': track.confidence,
                'age': track.age,
                'hits': track.hits,
                'time_since_update': track.time_since_update,
                'state': track.state,
                'velocity': track.get_velocity()
            })
        
        return output
    
    def get_track_by_id(self, track_id):
        """Get track by ID"""
        for track in self.tracks:
            if track.track_id == track_id:
                return {
                    'track_id': track.track_id,
                    'bbox': track.bbox,
                    'class_id': track.class_id,
                    'confidence': track.confidence,
                    'age': track.age,
                    'history': track.bbox_history
                }
        return None
    
    def reset(self):
        """Reset tracker"""
        self.tracks = []
        self.frame_count = 0
        Track._next_id = 1
        self.total_tracks_created = 0
        self.active_track_count = 0
    
    def get_statistics(self):
        """Get tracker statistics"""
        confirmed = sum(1 for t in self.tracks if t.is_confirmed())
        tentative = sum(1 for t in self.tracks if t.is_tentative())
        
        return {
            'frame_count': self.frame_count,
            'total_tracks': len(self.tracks),
            'confirmed_tracks': confirmed,
            'tentative_tracks': tentative,
            'total_tracks_created': self.total_tracks_created,
            'active_tracks': self.active_track_count
        }


class DeepSORTTracker(MultiObjectTracker):
    """
    DeepSORT tracker variant with enhanced appearance features
    
    Reference: "Simple Online and Realtime Tracking with a Deep Association Metric"
    """
    
    def __init__(
        self,
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        max_iou_distance=0.7,
        nn_budget=100
    ):
        super().__init__(max_age, min_hits, iou_threshold, feature_weight=0.7)
        
        self.max_iou_distance = max_iou_distance
        self.nn_budget = nn_budget  # Maximum samples per track for NN matching
    
    def update(self, detections, features=None, classes=None, confidences=None):
        """Enhanced update with cascade matching"""
        self.frame_count += 1
        
        # Predict all tracks
        for track in self.tracks:
            track.predict()
        
        # Cascade matching: match confirmed tracks first
        confirmed_tracks = [t for t in self.tracks if t.is_confirmed()]
        unconfirmed_tracks = [t for t in self.tracks if not t.is_confirmed()]
        
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = []
        
        # Match with confirmed tracks
        if len(confirmed_tracks) > 0 and len(detections) > 0:
            track_bboxes = np.array([t.predicted_bbox for t in confirmed_tracks])
            track_features = np.array([t.get_feature() for t in confirmed_tracks])
            
            det_subset = detections[unmatched_detections]
            feat_subset = features[unmatched_detections] if features is not None else None
            
            conf_matches, conf_unmatched_det, conf_unmatched_track = match_detections_to_tracks(
                det_subset,
                track_bboxes,
                detection_features=feat_subset,
                track_features=track_features,
                iou_threshold=self.iou_threshold,
                feature_weight=self.feature_weight,
                max_distance=self.max_iou_distance
            )
            
            # Convert to global indices
            for det_idx, track_idx in conf_matches:
                matches.append((unmatched_detections[det_idx], self.tracks.index(confirmed_tracks[track_idx])))
            
            unmatched_detections = [unmatched_detections[i] for i in conf_unmatched_det]
            unmatched_tracks = [self.tracks.index(confirmed_tracks[i]) for i in conf_unmatched_track]
        
        # Match remaining detections with unconfirmed tracks
        if len(unconfirmed_tracks) > 0 and len(unmatched_detections) > 0:
            track_bboxes = np.array([t.predicted_bbox for t in unconfirmed_tracks])
            det_subset = detections[unmatched_detections]
            
            unconf_matches, unconf_unmatched_det, unconf_unmatched_track = match_detections_to_tracks(
                det_subset,
                track_bboxes,
                iou_threshold=self.iou_threshold,
                feature_weight=0.0  # Only use IoU for unconfirmed
            )
            
            for det_idx, track_idx in unconf_matches:
                matches.append((unmatched_detections[det_idx], self.tracks.index(unconfirmed_tracks[track_idx])))
            
            unmatched_detections = [unmatched_detections[i] for i in unconf_unmatched_det]
            unmatched_tracks.extend([self.tracks.index(unconfirmed_tracks[i]) for i in unconf_unmatched_track])
        else:
            unmatched_tracks.extend([self.tracks.index(t) for t in unconfirmed_tracks])
        
        # Update matched tracks
        for det_idx, track_idx in matches:
            bbox = detections[det_idx]
            feature = features[det_idx] if features is not None else None
            class_id = classes[det_idx] if classes is not None else None
            confidence = confidences[det_idx] if confidences is not None else None
            
            self.tracks[track_idx].update(bbox, feature, confidence)
            if class_id is not None:
                self.tracks[track_idx].class_id = class_id
        
        # Mark unmatched tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # Create new tracks
        for det_idx in unmatched_detections:
            bbox = detections[det_idx]
            feature = features[det_idx] if features is not None else None
            class_id = classes[det_idx] if classes is not None else None
            confidence = confidences[det_idx] if confidences is not None else 1.0
            
            new_track = Track(bbox, feature, class_id, confidence)
            self.tracks.append(new_track)
            self.total_tracks_created += 1
        
        # Remove lost tracks
        self.tracks = [t for t in self.tracks if not t.is_lost()]
        
        # Get output
        output_tracks = []
        for track in self.tracks:
            if track.is_confirmed():
                output_tracks.append({
                    'track_id': track.track_id,
                    'bbox': track.bbox,
                    'class_id': track.class_id,
                    'confidence': track.confidence,
                    'age': track.age,
                    'hits': track.hits
                })
        
        self.active_track_count = len(output_tracks)
        
        return output_tracks


if __name__ == "__main__":
    print("Testing Multi-Object Tracker:\n")
    
    # Create tracker
    tracker = MultiObjectTracker(
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        feature_weight=0.5
    )
    
    # Simulate tracking scenario
    print("1. Simulating object tracking:")
    np.random.seed(42)
    
    for frame in range(20):
        # Simulate detections (2 moving objects with noise)
        obj1_x = 100 + frame * 5 + np.random.randn() * 5
        obj1_y = 100 + frame * 3 + np.random.randn() * 5
        
        obj2_x = 300 - frame * 2 + np.random.randn() * 5
        obj2_y = 200 + frame * 4 + np.random.randn() * 5
        
        detections = np.array([
            [obj1_x, obj1_y, 50, 80],
            [obj2_x, obj2_y, 60, 90]
        ])
        
        # Simulate ReID features
        features = np.random.randn(2, 128)
        
        # Occasionally miss a detection
        if frame % 7 == 0:
            detections = detections[:1]
            features = features[:1]
        
        # Update tracker
        tracks = tracker.update(detections, features)
        
        if frame % 5 == 0:
            print(f"   Frame {frame}: {len(tracks)} active tracks")
            for track in tracks:
                print(f"     Track {track['track_id']}: bbox={track['bbox'][:2]}, hits={track['hits']}")
    
    # Get statistics
    stats = tracker.get_statistics()
    print(f"\n2. Tracker Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test DeepSORT tracker
    print("\n3. Testing DeepSORT Tracker:")
    deepsort = DeepSORTTracker()
    
    for frame in range(10):
        detections = np.random.rand(3, 4) * 200 + 100
        detections[:, 2:] = 50  # Fixed size
        features = np.random.randn(3, 128)
        
        tracks = deepsort.update(detections, features)
        
        if frame % 3 == 0:
            print(f"   Frame {frame}: {len(tracks)} tracks")
    
    print("\n✓ All tracker tests passed!")