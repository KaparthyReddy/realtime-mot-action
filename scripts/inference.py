import os
import sys
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mot_action_net import MOTActionNet
from tracking.tracker import MultiObjectTracker
from utils.checkpoint import load_checkpoint
from evaluation.visualize import visualize_tracking


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on video')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video')
    parser.add_argument('--output', type=str, default='output.mp4',
                        help='Path to output video')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                        help='Confidence threshold for detections')
    parser.add_argument('--iou-threshold', type=float, default=0.3,
                        help='IoU threshold for tracking')
    parser.add_argument('--temporal-window', type=int, default=8,
                        help='Temporal window size')
    parser.add_argument('--show', action='store_true',
                        help='Display output in real-time')
    parser.add_argument('--save-frames', type=str, default=None,
                        help='Directory to save individual frames')
    
    return parser.parse_args()


def preprocess_frame(frame, target_size=(320, 320)):
    """Preprocess frame for model input"""
    # Resize
    frame_resized = cv2.resize(frame, target_size)
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    frame_normalized = frame_rgb.astype(np.float32) / 255.0
    
    # Convert to tensor (C, H, W)
    frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1)
    
    return frame_tensor


@torch.no_grad()
def run_inference(args):
    """Run inference on video"""
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint.get('config', {})
    
    model = MOTActionNet(
        num_classes=config.get('num_classes', 1),
        num_actions=config.get('num_actions', 10),
        temporal_window=args.temporal_window
    )
    
    load_checkpoint(args.checkpoint, model, device=device)
    model = model.to(device)
    model.eval()
    
    print("✓ Model loaded")
    
    # Initialize tracker
    tracker = MultiObjectTracker(
        max_age=30,
        min_hits=3,
        iou_threshold=args.iou_threshold
    )
    
    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {args.video}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # Setup frame saving
    if args.save_frames:
        os.makedirs(args.save_frames, exist_ok=True)
    
    # Frame buffer for temporal processing
    frame_buffer = []
    frame_count = 0
    
    print("\nProcessing video...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Preprocess frame
            frame_tensor = preprocess_frame(frame)
            frame_buffer.append(frame_tensor)
            
            # Process when we have enough frames
            if len(frame_buffer) >= args.temporal_window:
                # Prepare input batch
                input_frames = torch.stack(frame_buffer[-args.temporal_window:]).unsqueeze(0).to(device)
                
                # Run model
                predictions = model(input_frames, mode='all')
                
                # Process detections
                # TODO: Decode detection predictions properly
                # For now, using dummy detections
                detections = np.array([])
                features = None
                
                # Update tracker
                tracks = tracker.update(detections, features)
                
                # Visualize
                vis_frame = frame.copy()
                
                # Draw tracks
                for track in tracks:
                    bbox = track['bbox']
                    track_id = track['track_id']
                    
                    # Scale bbox to original resolution
                    x, y, w, h = bbox
                    x1 = int((x - w/2) * width / 320)
                    y1 = int((y - h/2) * height / 320)
                    x2 = int((x + w/2) * width / 320)
                    y2 = int((y + h/2) * height / 320)
                    
                    # Draw rectangle
                    color = (0, 255, 0)
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw ID
                    cv2.putText(
                        vis_frame,
                        f"ID: {track_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )
                
                # Add frame info
                cv2.putText(
                    vis_frame,
                    f"Frame: {frame_count}/{total_frames}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                # Write frame
                out.write(vis_frame)
                
                # Save frame if requested
                if args.save_frames:
                    frame_path = os.path.join(args.save_frames, f"frame_{frame_count:06d}.jpg")
                    cv2.imwrite(frame_path, vis_frame)
                
                # Show frame if requested
                if args.show:
                    cv2.imshow('Tracking', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            # Progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"  Progress: {progress:.1f}% ({frame_count}/{total_frames})")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        out.release()
        if args.show:
            cv2.destroyAllWindows()
    
    print(f"\n✓ Output saved to: {args.output}")
    if args.save_frames:
        print(f"✓ Frames saved to: {args.save_frames}")


def main():
    args = parse_args()
    
    # Check if video exists
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # Run inference
    run_inference(args)


if __name__ == '__main__':
    main()