import numpy as np
from collections import defaultdict


class MOTMetrics:
    """
    Multi-Object Tracking Metrics
    
    Computes standard MOT evaluation metrics:
    - MOTA: Multi-Object Tracking Accuracy
    - MOTP: Multi-Object Tracking Precision
    - IDF1: ID F1 Score
    - MT: Mostly Tracked targets
    - ML: Mostly Lost targets
    - FP: False Positives
    - FN: False Negatives
    - ID Switches
    """
    
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        """Reset all counters"""
        self.num_frames = 0
        self.num_gt = 0
        self.num_det = 0
        self.num_matches = 0
        self.num_false_positives = 0
        self.num_misses = 0
        self.num_switches = 0
        self.total_iou = 0.0
        
        # Track-level statistics
        self.track_gt_ids = set()
        self.track_ratios = defaultdict(lambda: {'tracked': 0, 'total': 0})
        
        # For IDF1
        self.idtp = 0  # ID True Positives
        self.idfp = 0  # ID False Positives
        self.idfn = 0  # ID False Negatives
        
        # Track last matched IDs for switch detection
        self.prev_matches = {}
    
    def update(self, gt_boxes, gt_ids, pred_boxes, pred_ids):
        """
        Update metrics with frame data
        
        Args:
            gt_boxes: (N, 4) ground truth boxes [x, y, w, h]
            gt_ids: (N,) ground truth IDs
            pred_boxes: (M, 4) predicted boxes [x, y, w, h]
            pred_ids: (M,) predicted IDs
        """
        self.num_frames += 1
        self.num_gt += len(gt_boxes)
        self.num_det += len(pred_boxes)
        
        # Track ground truth IDs
        for gt_id in gt_ids:
            self.track_gt_ids.add(gt_id)
            self.track_ratios[gt_id]['total'] += 1
        
        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            self.num_misses += len(gt_boxes)
            self.num_false_positives += len(pred_boxes)
            return
        
        # Compute IoU matrix
        iou_matrix = self._compute_iou_matrix(gt_boxes, pred_boxes)
        
        # Match predictions to ground truth
        matches, unmatched_gt, unmatched_pred = self._match(iou_matrix)
        
        # Update counters
        self.num_matches += len(matches)
        self.num_misses += len(unmatched_gt)
        self.num_false_positives += len(unmatched_pred)
        
        # Compute MOTP (average IoU of matches)
        for gt_idx, pred_idx in matches:
            self.total_iou += iou_matrix[gt_idx, pred_idx]
            
            # Track ratio for MT/ML
            gt_id = gt_ids[gt_idx]
            self.track_ratios[gt_id]['tracked'] += 1
            
            # Check for ID switch
            pred_id = pred_ids[pred_idx]
            if gt_id in self.prev_matches:
                if self.prev_matches[gt_id] != pred_id:
                    self.num_switches += 1
            self.prev_matches[gt_id] = pred_id
        
        # IDF1 calculation
        self.idtp += len(matches)
        self.idfp += len(unmatched_pred)
        self.idfn += len(unmatched_gt)
    
    def compute(self):
        """Compute final metrics"""
        if self.num_gt == 0:
            return {}
        
        # MOTA: Multi-Object Tracking Accuracy
        mota = 1 - (self.num_false_positives + self.num_misses + self.num_switches) / self.num_gt
        
        # MOTP: Multi-Object Tracking Precision (average IoU)
        motp = self.total_iou / self.num_matches if self.num_matches > 0 else 0
        
        # IDF1: ID F1 Score
        idf1 = 2 * self.idtp / (2 * self.idtp + self.idfp + self.idfn) if (self.idtp + self.idfp + self.idfn) > 0 else 0
        
        # Precision and Recall
        precision = self.num_matches / self.num_det if self.num_det > 0 else 0
        recall = self.num_matches / self.num_gt if self.num_gt > 0 else 0
        
        # MT (Mostly Tracked): tracked for at least 80% of their lifetime
        # ML (Mostly Lost): tracked for at most 20% of their lifetime
        mt = sum(1 for stats in self.track_ratios.values() if stats['tracked'] / stats['total'] >= 0.8)
        ml = sum(1 for stats in self.track_ratios.values() if stats['tracked'] / stats['total'] <= 0.2)
        
        return {
            'MOTA': mota * 100,
            'MOTP': motp * 100,
            'IDF1': idf1 * 100,
            'MT': mt,
            'ML': ml,
            'FP': self.num_false_positives,
            'FN': self.num_misses,
            'ID_Sw': self.num_switches,
            'Precision': precision * 100,
            'Recall': recall * 100,
            'Num_GT': self.num_gt,
            'Num_Det': self.num_det
        }
    
    def _compute_iou_matrix(self, boxes1, boxes2):
        """Compute IoU matrix between two sets of boxes"""
        N = len(boxes1)
        M = len(boxes2)
        iou_matrix = np.zeros((N, M))
        
        for i in range(N):
            for j in range(M):
                iou_matrix[i, j] = self._iou(boxes1[i], boxes2[j])
        
        return iou_matrix
    
    def _iou(self, box1, box2):
        """Compute IoU between two boxes [x, y, w, h]"""
        x1_1, y1_1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
        x2_1, y2_1 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
        
        x1_2, y1_2 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
        x2_2, y2_2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        inter_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / (union_area + 1e-7)
    
    def _match(self, iou_matrix):
        """Match predictions to ground truth using greedy matching"""
        matches = []
        unmatched_gt = set(range(iou_matrix.shape[0]))
        unmatched_pred = set(range(iou_matrix.shape[1]))
        
        # Greedy matching based on IoU
        while iou_matrix.max() > self.iou_threshold:
            i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            
            if i in unmatched_gt and j in unmatched_pred:
                matches.append((i, j))
                unmatched_gt.remove(i)
                unmatched_pred.remove(j)
                
                iou_matrix[i, :] = 0
                iou_matrix[:, j] = 0
            else:
                break
        
        return matches, list(unmatched_gt), list(unmatched_pred)


def compute_mot_metrics(gt_data, pred_data, iou_threshold=0.5):
    """
    Compute MOT metrics from ground truth and predictions
    
    Args:
        gt_data: Dict mapping frame_id to dict with 'boxes' and 'ids'
        pred_data: Dict mapping frame_id to dict with 'boxes' and 'ids'
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary of metrics
    """
    metrics = MOTMetrics(iou_threshold=iou_threshold)
    
    # Get all frame IDs
    all_frames = sorted(set(gt_data.keys()) | set(pred_data.keys()))
    
    for frame_id in all_frames:
        gt_boxes = gt_data.get(frame_id, {}).get('boxes', np.array([]))
        gt_ids = gt_data.get(frame_id, {}).get('ids', np.array([]))
        
        pred_boxes = pred_data.get(frame_id, {}).get('boxes', np.array([]))
        pred_ids = pred_data.get(frame_id, {}).get('ids', np.array([]))
        
        metrics.update(gt_boxes, gt_ids, pred_boxes, pred_ids)
    
    return metrics.compute()


if __name__ == "__main__":
    print("Testing MOT Metrics:\n")
    
    # Create dummy tracking data
    np.random.seed(42)
    
    gt_data = {}
    pred_data = {}
    
    # Simulate 50 frames with 3 objects
    for frame in range(50):
        # Ground truth
        gt_boxes = np.array([
            [100 + frame * 2, 100 + frame, 50, 80],
            [300 - frame, 200 + frame * 1.5, 60, 90],
            [500, 150 + frame * 0.5, 55, 85]
        ])
        gt_ids = np.array([1, 2, 3])
        
        gt_data[frame] = {'boxes': gt_boxes, 'ids': gt_ids}
        
        # Predictions (with some noise and occasional misses)
        if frame % 10 != 0:  # Occasionally miss detections
            pred_boxes = gt_boxes + np.random.randn(3, 4) * 5
            pred_ids = gt_ids.copy()
            
            # Simulate ID switch at frame 25
            if frame == 25:
                pred_ids[0] = 4  # ID switch for object 1
            
            pred_data[frame] = {'boxes': pred_boxes, 'ids': pred_ids}
    
    # Compute metrics
    metrics = compute_mot_metrics(gt_data, pred_data)
    
    print("MOT Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")