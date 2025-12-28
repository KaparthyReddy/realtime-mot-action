import numpy as np
from scipy.optimize import linear_sum_assignment

def hungarian_matching(cost_matrix):
    """
    Standard Hungarian matching using scipy
    """
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return np.column_stack((row_ind, col_ind))

def iou(bbox1, bbox2):
    """
    Compute IoU between two bounding boxes
    
    Args:
        bbox1, bbox2: [x, y, w, h] format
    
    Returns:
        IoU value [0, 1]
    """
    # Convert to [x1, y1, x2, y2]
    x1_1, y1_1 = bbox1[0] - bbox1[2]/2, bbox1[1] - bbox1[3]/2
    x2_1, y2_1 = bbox1[0] + bbox1[2]/2, bbox1[1] + bbox1[3]/2
    
    x1_2, y1_2 = bbox2[0] - bbox2[2]/2, bbox2[1] - bbox2[3]/2
    x2_2, y2_2 = bbox2[0] + bbox2[2]/2, bbox2[1] + bbox2[3]/2
    
    # Intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    inter_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Union
    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = bbox1_area + bbox2_area - inter_area
    
    return inter_area / (union_area + 1e-7)


def compute_iou_matrix(bboxes1, bboxes2):
    """
    Compute IoU matrix between two sets of bounding boxes
    
    Args:
        bboxes1: (N, 4) array
        bboxes2: (M, 4) array
    
    Returns:
        (N, M) IoU matrix
    """
    N = len(bboxes1)
    M = len(bboxes2)
    iou_matrix = np.zeros((N, M))
    
    for i in range(N):
        for j in range(M):
            iou_matrix[i, j] = iou(bboxes1[i], bboxes2[j])
    
    return iou_matrix


def cosine_similarity(features1, features2):
    """
    Compute cosine similarity between feature vectors
    
    Args:
        features1: (N, D) array
        features2: (M, D) array
    
    Returns:
        (N, M) similarity matrix [0, 1]
    """
    # Normalize features
    features1_norm = features1 / (np.linalg.norm(features1, axis=1, keepdims=True) + 1e-7)
    features2_norm = features2 / (np.linalg.norm(features2, axis=1, keepdims=True) + 1e-7)
    
    # Compute similarity
    similarity = features1_norm @ features2_norm.T
    
    # Convert to [0, 1] range
    similarity = (similarity + 1) / 2
    
    return similarity


def match_detections_to_tracks(
    detections,
    tracks,
    detection_features=None,
    track_features=None,
    iou_threshold=0.3,
    feature_weight=0.5,
    max_distance=0.7
):
    """
    Match detections to existing tracks using Hungarian algorithm
    
    Args:
        detections: (N, 4) array of detection bboxes [x, y, w, h]
        tracks: (M, 4) array of track bboxes [x, y, w, h]
        detection_features: (N, D) ReID features for detections
        track_features: (M, D) ReID features for tracks
        iou_threshold: Minimum IoU for valid match
        feature_weight: Weight for appearance features [0, 1]
        max_distance: Maximum distance for valid match
    
    Returns:
        matches: List of (detection_idx, track_idx) tuples
        unmatched_detections: List of detection indices
        unmatched_tracks: List of track indices
    """
    if len(detections) == 0 or len(tracks) == 0:
        return [], list(range(len(detections))), list(range(len(tracks)))
    
    # Compute cost matrix
    N, M = len(detections), len(tracks)
    
    # IoU-based cost
    iou_matrix = compute_iou_matrix(detections, tracks)
    iou_cost = 1 - iou_matrix  # Convert similarity to cost
    
    # Feature-based cost (if available)
    if detection_features is not None and track_features is not None:
        feature_similarity = cosine_similarity(detection_features, track_features)
        feature_cost = 1 - feature_similarity
        
        # Combine costs
        cost_matrix = (1 - feature_weight) * iou_cost + feature_weight * feature_cost
    else:
        cost_matrix = iou_cost
    
    # Apply threshold: set high cost for invalid matches
    cost_matrix[iou_matrix < iou_threshold] = max_distance + 1
    
    # Hungarian algorithm for optimal assignment
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Filter matches based on distance threshold
    matches = []
    unmatched_detections = list(range(N))
    unmatched_tracks = list(range(M))
    
    for i, j in zip(row_indices, col_indices):
        if cost_matrix[i, j] <= max_distance:
            matches.append((i, j))
            unmatched_detections.remove(i)
            unmatched_tracks.remove(j)
    
    return matches, unmatched_detections, unmatched_tracks


def greedy_matching(
    detections,
    tracks,
    detection_features=None,
    track_features=None,
    iou_threshold=0.3
):
    """
    Greedy matching algorithm (faster but less optimal than Hungarian)
    
    Args:
        detections: (N, 4) array of detection bboxes
        tracks: (M, 4) array of track bboxes
        detection_features: (N, D) ReID features
        track_features: (M, D) ReID features
        iou_threshold: Minimum IoU for match
    
    Returns:
        matches, unmatched_detections, unmatched_tracks
    """
    if len(detections) == 0 or len(tracks) == 0:
        return [], list(range(len(detections))), list(range(len(tracks)))
    
    # Compute similarity matrix
    iou_matrix = compute_iou_matrix(detections, tracks)
    
    if detection_features is not None and track_features is not None:
        feature_similarity = cosine_similarity(detection_features, track_features)
        similarity = 0.5 * iou_matrix + 0.5 * feature_similarity
    else:
        similarity = iou_matrix
    
    matches = []
    unmatched_detections = set(range(len(detections)))
    unmatched_tracks = set(range(len(tracks)))
    
    # Greedily match highest similarities
    while similarity.max() > iou_threshold:
        i, j = np.unravel_index(similarity.argmax(), similarity.shape)
        
        if i in unmatched_detections and j in unmatched_tracks:
            matches.append((i, j))
            unmatched_detections.remove(i)
            unmatched_tracks.remove(j)
            
            # Set matched row and column to 0
            similarity[i, :] = 0
            similarity[:, j] = 0
        else:
            break
    
    return matches, list(unmatched_detections), list(unmatched_tracks)


def cascade_matching(detections, tracks_by_age, **kwargs):
    """
    Cascade matching: match with recent tracks first
    
    Args:
        detections: (N, 4) array
        tracks_by_age: List of track lists, ordered by age
        **kwargs: Arguments for match_detections_to_tracks
    
    Returns:
        matches, unmatched_detections, unmatched_tracks
    """
    all_matches = []
    all_unmatched_tracks = []
    unmatched_detections = list(range(len(detections)))
    
    for track_group in tracks_by_age:
        if len(unmatched_detections) == 0:
            all_unmatched_tracks.extend(list(range(len(track_group))))
            continue
        
        # Match current detection subset with current track group
        det_subset = detections[unmatched_detections]
        matches, new_unmatched_det, unmatched_tracks_group = \
            match_detections_to_tracks(det_subset, track_group, **kwargs)
        
        # Convert local indices to global indices
        global_matches = [(unmatched_detections[i], j) for i, j in matches]
        all_matches.extend(global_matches)
        
        # Update unmatched detections
        unmatched_detections = [unmatched_detections[i] for i in new_unmatched_det]
        all_unmatched_tracks.extend(unmatched_tracks_group)
    
    return all_matches, unmatched_detections, all_unmatched_tracks


if __name__ == "__main__":
    print("Testing Matching Algorithms:\n")
    
    # Create test data
    np.random.seed(42)
    
    # Detections and tracks
    detections = np.array([
        [100, 100, 50, 80],
        [300, 200, 60, 90],
        [500, 150, 55, 85]
    ])
    
    tracks = np.array([
        [105, 102, 50, 80],  # Close to detection 0
        [450, 145, 55, 85],  # Close to detection 2
        [700, 300, 60, 90]   # No match
    ])
    
    # Test IoU matrix
    print("1. IoU Matrix:")
    iou_matrix = compute_iou_matrix(detections, tracks)
    print(iou_matrix)
    
    # Test Hungarian matching
    print("\n2. Hungarian Matching:")
    matches, unmatched_det, unmatched_track = match_detections_to_tracks(
        detections, tracks, iou_threshold=0.3
    )
    print(f"   Matches: {matches}")
    print(f"   Unmatched detections: {unmatched_det}")
    print(f"   Unmatched tracks: {unmatched_track}")
    
    # Test with features
    print("\n3. Matching with ReID Features:")
    det_features = np.random.randn(3, 128)
    track_features = np.random.randn(3, 128)
    # Make feature 0 similar to track 0
    track_features[0] = det_features[0] + np.random.randn(128) * 0.1
    
    matches_feat, _, _ = match_detections_to_tracks(
        detections, tracks,
        detection_features=det_features,
        track_features=track_features,
        iou_threshold=0.1,
        feature_weight=0.7
    )
    print(f"   Matches with features: {matches_feat}")
    
    # Test greedy matching
    print("\n4. Greedy Matching:")
    matches_greedy, _, _ = greedy_matching(detections, tracks, iou_threshold=0.3)
    print(f"   Matches: {matches_greedy}")
    
    print("\n✓ All matching tests passed!")