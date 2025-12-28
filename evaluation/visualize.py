import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import seaborn as sns


def visualize_detections(image, boxes, labels=None, scores=None, class_names=None, figsize=(12, 8)):
    """
    Visualize bounding box detections on an image
    
    Args:
        image: (H, W, 3) RGB image
        boxes: (N, 4) boxes in [x, y, w, h] format
        labels: (N,) class labels
        scores: (N,) confidence scores
        class_names: List of class names
        figsize: Figure size
    """
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)
    
    # Color palette
    colors = plt.cm.get_cmap('tab10')
    
    for i, box in enumerate(boxes):
        x, y, w, h = box
        
        # Convert center format to corner format
        x1 = x - w / 2
        y1 = y - h / 2
        
        # Determine color
        if labels is not None:
            color = colors(labels[i] % 10)
        else:
            color = 'green'
        
        # Draw rectangle
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        label_text = ""
        if labels is not None and class_names is not None:
            label_text = class_names[labels[i]]
        if scores is not None:
            label_text += f" {scores[i]:.2f}"
        
        if label_text:
            ax.text(
                x1, y1 - 5,
                label_text,
                color='white',
                fontsize=10,
                bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=2)
            )
    
    ax.axis('off')
    plt.tight_layout()
    return fig


def visualize_tracking(image, tracks, class_names=None, show_ids=True, show_trajectories=False, figsize=(12, 8)):
    """
    Visualize tracked objects with IDs
    
    Args:
        image: (H, W, 3) RGB image
        tracks: List of track dicts with 'bbox', 'track_id', 'class_id'
        class_names: List of class names
        show_ids: Whether to show track IDs
        show_trajectories: Whether to show past trajectories
        figsize: Figure size
    """
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)
    
    # Color palette (consistent colors per track ID)
    colors = plt.cm.get_cmap('tab20')
    
    for track in tracks:
        bbox = track['bbox']
        track_id = track['track_id']
        x, y, w, h = bbox
        
        # Convert to corner format
        x1 = x - w / 2
        y1 = y - h / 2
        
        # Color based on track ID
        color = colors(track_id % 20)
        
        # Draw rectangle
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=3,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        label_text = f"ID: {track_id}"
        if 'class_id' in track and class_names is not None:
            label_text = f"{class_names[track['class_id']]} {label_text}"
        
        if show_ids:
            ax.text(
                x1, y1 - 5,
                label_text,
                color='white',
                fontsize=11,
                fontweight='bold',
                bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', pad=3)
            )
        
        # Draw trajectory
        if show_trajectories and 'history' in track:
            history = track['history']
            if len(history) > 1:
                centers = np.array([[h[0], h[1]] for h in history])
                ax.plot(centers[:, 0], centers[:, 1], color=color, linewidth=2, alpha=0.6)
    
    ax.axis('off')
    plt.tight_layout()
    return fig


def plot_confusion_matrix(confusion_matrix, class_names, normalize=False, figsize=(10, 8)):
    """
    Plot confusion matrix
    
    Args:
        confusion_matrix: (N, N) confusion matrix
        class_names: List of class names
        normalize: Whether to normalize
        figsize: Figure size
    """
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / (
            confusion_matrix.sum(axis=1)[:, np.newaxis] + 1e-7
        )
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_metrics(metrics_history, metric_names=None, figsize=(15, 5)):
    """
    Plot training metrics over time
    
    Args:
        metrics_history: Dict mapping metric names to lists of values
        metric_names: List of metric names to plot (default: all)
        figsize: Figure size
    """
    if metric_names is None:
        metric_names = list(metrics_history.keys())
    
    num_metrics = len(metric_names)
    fig, axes = plt.subplots(1, num_metrics, figsize=figsize)
    
    if num_metrics == 1:
        axes = [axes]
    
    for ax, metric_name in zip(axes, metric_names):
        values = metrics_history[metric_name]
        epochs = range(1, len(values) + 1)
        
        ax.plot(epochs, values, marker='o', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_mot_metrics_comparison(metrics_dict, figsize=(14, 6)):
    """
    Compare MOT metrics across different methods
    
    Args:
        metrics_dict: Dict mapping method names to metric dicts
        figsize: Figure size
    """
    methods = list(metrics_dict.keys())
    metric_names = ['MOTA', 'MOTP', 'IDF1', 'Precision', 'Recall']
    
    # Prepare data
    data = {metric: [] for metric in metric_names}
    for method in methods:
        for metric in metric_names:
            data[metric].append(metrics_dict[method].get(metric, 0))
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Bar chart
    x = np.arange(len(methods))
    width = 0.15
    
    for i, metric in enumerate(metric_names):
        axes[0].bar(x + i * width, data[metric], width, label=metric)
    
    axes[0].set_xlabel('Method', fontsize=12)
    axes[0].set_ylabel('Score (%)', fontsize=12)
    axes[0].set_title('MOT Metrics Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x + width * 2)
    axes[0].set_xticklabels(methods, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, axis='y', alpha=0.3)
    
    # Radar chart
    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]
    
    ax = plt.subplot(122, projection='polar')
    
    for method in methods:
        values = [metrics_dict[method].get(m, 0) for m in metric_names]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=method)
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 100)
    ax.set_title('Metric Comparison (Radar)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    return fig


def save_tracking_video(frames, tracks_per_frame, output_path, fps=30, class_names=None):
    """
    Save tracking visualization as video
    
    Args:
        frames: List of (H, W, 3) images
        tracks_per_frame: List of track lists (one per frame)
        output_path: Path to save video
        fps: Frames per second
        class_names: List of class names
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    def update(frame_idx):
        ax.clear()
        ax.imshow(frames[frame_idx])
        
        tracks = tracks_per_frame[frame_idx]
        colors = plt.cm.get_cmap('tab20')
        
        for track in tracks:
            bbox = track['bbox']
            track_id = track['track_id']
            x, y, w, h = bbox
            
            x1 = x - w / 2
            y1 = y - h / 2
            
            color = colors(track_id % 20)
            
            rect = patches.Rectangle(
                (x1, y1), w, h,
                linewidth=3,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)
            
            label_text = f"ID: {track_id}"
            if 'class_id' in track and class_names is not None:
                label_text = f"{class_names[track['class_id']]} {label_text}"
            
            ax.text(
                x1, y1 - 5,
                label_text,
                color='white',
                fontsize=11,
                fontweight='bold',
                bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', pad=3)
            )
        
        ax.axis('off')
        ax.set_title(f"Frame {frame_idx + 1}/{len(frames)}", fontsize=14, fontweight='bold')
    
    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000//fps)
    anim.save(output_path, writer='pillow', fps=fps)
    plt.close()
    
    print(f"Video saved to {output_path}")


if __name__ == "__main__":
    print("Testing Visualization Functions:\n")
    
    # Create dummy image
    np.random.seed(42)
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test detection visualization
    print("1. Testing detection visualization...")
    boxes = np.array([
        [100, 100, 80, 120],
        [300, 200, 90, 110],
        [500, 150, 85, 125]
    ])
    labels = np.array([0, 1, 2])
    scores = np.array([0.95, 0.87, 0.92])
    class_names = ['Person', 'Car', 'Bicycle']
    
    fig = visualize_detections(image, boxes, labels, scores, class_names)
    plt.savefig('test_detection.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("   Saved to test_detection.png")
    
    # Test tracking visualization
    print("\n2. Testing tracking visualization...")
    tracks = [
        {'bbox': [100, 100, 80, 120], 'track_id': 1, 'class_id': 0},
        {'bbox': [300, 200, 90, 110], 'track_id': 2, 'class_id': 1},
        {'bbox': [500, 150, 85, 125], 'track_id': 3, 'class_id': 2}
    ]
    
    fig = visualize_tracking(image, tracks, class_names, show_ids=True)
    plt.savefig('test_tracking.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("   Saved to test_tracking.png")
    
    # Test confusion matrix
    print("\n3. Testing confusion matrix...")
    conf_matrix = np.random.randint(0, 100, (5, 5))
    fig = plot_confusion_matrix(conf_matrix, [f"Class_{i}" for i in range(5)])
    plt.savefig('test_confusion_matrix.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("   Saved to test_confusion_matrix.png")
    
    # Test metrics plot
    print("\n4. Testing metrics plot...")
    metrics_history = {
        'Loss': [2.5, 2.0, 1.7, 1.5, 1.3, 1.2, 1.1],
        'Accuracy': [60, 68, 73, 77, 80, 82, 84],
        'F1': [58, 66, 71, 75, 78, 80, 82]
    }
    fig = plot_metrics(metrics_history)
    plt.savefig('test_metrics.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("   Saved to test_metrics.png")
    
    print("\n✓ All visualization tests completed!")