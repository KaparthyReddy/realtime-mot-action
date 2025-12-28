import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)


class ActionMetrics:
    """
    Action Recognition Metrics
    
    Computes:
    - Accuracy
    - Precision, Recall, F1 per class
    - Confusion Matrix
    - Top-k Accuracy
    """
    
    def __init__(self, num_classes, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """Reset all stored predictions"""
        self.predictions = []
        self.targets = []
        self.confidences = []
    
    def update(self, predictions, targets, confidences=None):
        """
        Update with batch predictions
        
        Args:
            predictions: (N,) predicted class indices
            targets: (N,) ground truth class indices
            confidences: (N, num_classes) prediction probabilities (optional)
        """
        if isinstance(predictions, np.ndarray):
            self.predictions.extend(predictions.tolist())
        else:
            self.predictions.extend(predictions)
        
        if isinstance(targets, np.ndarray):
            self.targets.extend(targets.tolist())
        else:
            self.targets.extend(targets)
        
        if confidences is not None:
            if isinstance(confidences, np.ndarray):
                self.confidences.extend(confidences.tolist())
            else:
                self.confidences.extend(confidences)
    
    def compute(self):
        """Compute all metrics"""
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Overall accuracy
        accuracy = accuracy_score(targets, preds) * 100
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, preds, average=None, zero_division=0
        )
        
        # Macro averages
        precision_macro = np.mean(precision) * 100
        recall_macro = np.mean(recall) * 100
        f1_macro = np.mean(f1) * 100
        
        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            targets, preds, average='weighted', zero_division=0
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(targets, preds)
        
        # Top-k accuracy (if confidences available)
        top5_acc = None
        if len(self.confidences) > 0:
            top5_acc = self._compute_topk_accuracy(
                np.array(self.confidences), targets, k=5
            ) * 100
        
        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted * 100,
            'recall_weighted': recall_weighted * 100,
            'f1_weighted': f1_weighted * 100,
            'confusion_matrix': conf_matrix,
            'per_class': {
                'precision': precision * 100,
                'recall': recall * 100,
                'f1': f1 * 100,
                'support': support
            }
        }
        
        if top5_acc is not None:
            metrics['top5_accuracy'] = top5_acc
        
        return metrics
    
    def _compute_topk_accuracy(self, confidences, targets, k=5):
        """Compute top-k accuracy"""
        if confidences.shape[1] < k:
            k = confidences.shape[1]
        
        # Get top-k predictions
        topk_preds = np.argsort(confidences, axis=1)[:, -k:]
        
        # Check if target is in top-k
        correct = 0
        for i, target in enumerate(targets):
            if target in topk_preds[i]:
                correct += 1
        
        return correct / len(targets)
    
    def get_classification_report(self):
        """Get detailed classification report"""
        return classification_report(
            self.targets,
            self.predictions,
            target_names=self.class_names,
            zero_division=0
        )
    
    def get_confusion_matrix(self, normalize=False):
        """
        Get confusion matrix
        
        Args:
            normalize: If True, normalize confusion matrix
        """
        conf_matrix = confusion_matrix(self.targets, self.predictions)
        
        if normalize:
            conf_matrix = conf_matrix.astype('float') / (conf_matrix.sum(axis=1)[:, np.newaxis] + 1e-7)
        
        return conf_matrix


def compute_action_metrics(predictions, targets, num_classes, class_names=None):
    """
    Compute action recognition metrics
    
    Args:
        predictions: (N,) or (N, num_classes) predictions
        targets: (N,) ground truth labels
        num_classes: Number of action classes
        class_names: List of class names
    
    Returns:
        Dictionary of metrics
    """
    metrics = ActionMetrics(num_classes, class_names)
    
    # Handle probability predictions
    if predictions.ndim == 2:
        confidences = predictions
        predictions = np.argmax(predictions, axis=1)
        metrics.update(predictions, targets, confidences)
    else:
        metrics.update(predictions, targets)
    
    return metrics.compute()


class TemporalActionMetrics:
    """
    Metrics for temporal action detection/segmentation
    
    For frame-level action predictions in videos
    """
    
    def __init__(self, num_classes, overlap_threshold=0.5):
        self.num_classes = num_classes
        self.overlap_threshold = overlap_threshold
        self.reset()
    
    def reset(self):
        """Reset metrics"""
        self.frame_predictions = []
        self.frame_targets = []
    
    def update(self, predictions, targets):
        """
        Update with frame-level predictions
        
        Args:
            predictions: (T,) frame-level predictions for video
            targets: (T,) frame-level ground truth
        """
        self.frame_predictions.append(predictions)
        self.frame_targets.append(targets)
    
    def compute(self):
        """Compute temporal metrics"""
        all_preds = np.concatenate(self.frame_predictions)
        all_targets = np.concatenate(self.frame_targets)
        
        # Frame-level accuracy
        frame_accuracy = accuracy_score(all_targets, all_preds) * 100
        
        # Per-class frame accuracy
        per_class_acc = []
        for c in range(self.num_classes):
            mask = all_targets == c
            if mask.sum() > 0:
                acc = accuracy_score(all_targets[mask], all_preds[mask]) * 100
                per_class_acc.append(acc)
            else:
                per_class_acc.append(0.0)
        
        # Segment-level metrics (detect action boundaries)
        # TODO: Implement temporal IoU for action segments
        
        return {
            'frame_accuracy': frame_accuracy,
            'per_class_frame_accuracy': per_class_acc,
            'mean_per_class_accuracy': np.mean(per_class_acc)
        }


if __name__ == "__main__":
    print("Testing Action Metrics:\n")
    
    # Create dummy action data
    np.random.seed(42)
    
    num_classes = 10
    num_samples = 1000
    
    # Simulate predictions with some noise
    targets = np.random.randint(0, num_classes, num_samples)
    
    # Predictions with 80% accuracy
    predictions = targets.copy()
    noise_indices = np.random.choice(num_samples, size=int(num_samples * 0.2), replace=False)
    predictions[noise_indices] = np.random.randint(0, num_classes, len(noise_indices))
    
    # Confidence scores
    confidences = np.random.rand(num_samples, num_classes)
    for i, pred in enumerate(predictions):
        confidences[i, pred] += 0.5  # Boost correct class
    confidences = confidences / confidences.sum(axis=1, keepdims=True)
    
    # Compute metrics
    class_names = [f"Action_{i}" for i in range(num_classes)]
    metrics = compute_action_metrics(confidences, targets, num_classes, class_names)
    
    print("Action Recognition Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.2f}%")
    print(f"  Top-5 Accuracy: {metrics['top5_accuracy']:.2f}%")
    print(f"  Precision (macro): {metrics['precision_macro']:.2f}%")
    print(f"  Recall (macro): {metrics['recall_macro']:.2f}%")
    print(f"  F1 (macro): {metrics['f1_macro']:.2f}%")
    
    print("\nPer-class metrics:")
    for i in range(min(3, num_classes)):
        print(f"  {class_names[i]}:")
        print(f"    Precision: {metrics['per_class']['precision'][i]:.2f}%")
        print(f"    Recall: {metrics['per_class']['recall'][i]:.2f}%")
        print(f"    F1: {metrics['per_class']['f1'][i]:.2f}%")
        print(f"    Support: {metrics['per_class']['support'][i]}")
    
    print("\nConfusion Matrix shape:", metrics['confusion_matrix'].shape)
    
    # Test temporal metrics
    print("\n" + "="*60)
    print("\nTesting Temporal Action Metrics:")
    
    temporal_metrics = TemporalActionMetrics(num_classes=10)
    
    # Simulate 5 videos with frame-level predictions
    for _ in range(5):
        video_length = np.random.randint(50, 150)
        frame_targets = np.random.randint(0, 10, video_length)
        frame_preds = frame_targets.copy()
        # Add 20% error
        noise_idx = np.random.choice(video_length, size=int(video_length * 0.2), replace=False)
        frame_preds[noise_idx] = np.random.randint(0, 10, len(noise_idx))
        
        temporal_metrics.update(frame_preds, frame_targets)
    
    temp_results = temporal_metrics.compute()
    print(f"  Frame Accuracy: {temp_results['frame_accuracy']:.2f}%")
    print(f"  Mean Per-Class Accuracy: {temp_results['mean_per_class_accuracy']:.2f}%")