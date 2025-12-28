import os
import sys
import argparse
import urllib.request
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file from URL with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def parse_args():
    parser = argparse.ArgumentParser(description='Download datasets')
    
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['mot17', 'mot20', 'ava', 'kinetics', 'dummy', 'all'],
                        help='Dataset to download')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory to save data')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip if dataset already exists')
    
    return parser.parse_args()


def download_mot17(data_dir):
    """Download MOT17 dataset"""
    print("\nDownloading MOT17 dataset...")
    print("Note: MOT17 requires registration at https://motchallenge.net/")
    print("Please download manually and place in", data_dir)
    
    mot17_dir = os.path.join(data_dir, 'MOT17')
    os.makedirs(mot17_dir, exist_ok=True)
    
    print(f"✓ MOT17 directory created at: {mot17_dir}")
    print("  Please download and extract the dataset there.")


def download_mot20(data_dir):
    """Download MOT20 dataset"""
    print("\nDownloading MOT20 dataset...")
    print("Note: MOT20 requires registration at https://motchallenge.net/")
    print("Please download manually and place in", data_dir)
    
    mot20_dir = os.path.join(data_dir, 'MOT20')
    os.makedirs(mot20_dir, exist_ok=True)
    
    print(f"✓ MOT20 directory created at: {mot20_dir}")
    print("  Please download and extract the dataset there.")


def download_ava(data_dir):
    """Download AVA dataset annotations"""
    print("\nDownloading AVA dataset...")
    print("Note: AVA requires downloading videos separately")
    
    ava_dir = os.path.join(data_dir, 'AVA')
    os.makedirs(ava_dir, exist_ok=True)
    
    # Download annotations (these are publicly available)
    print("  AVA annotations are typically accessed via official channels")
    print("  Please visit: https://research.google.com/ava/download.html")
    
    print(f"\n✓ AVA directory created at: {ava_dir}")
    print("  Please download annotations and videos from the official website")


def create_dummy_data(data_dir):
    """Create dummy data for testing"""
    print("\nCreating dummy test data...")
    
    dummy_dir = os.path.join(data_dir, 'dummy')
    os.makedirs(dummy_dir, exist_ok=True)
    
    # Create dummy video frames
    import numpy as np
    import cv2
    
    video_dir = os.path.join(dummy_dir, 'videos')
    os.makedirs(video_dir, exist_ok=True)
    
    # Create a simple test video
    print("  Creating test video...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(video_dir, 'test_video.mp4')
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
    
    for i in range(100):
        # Create frame with moving rectangles
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Draw multiple moving objects
        # Object 1: Moving right
        x1 = int(50 + i * 3)
        cv2.rectangle(frame, (x1, 100), (x1+60, 180), (0, 255, 0), -1)
        cv2.putText(frame, "1", (x1+20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Object 2: Moving down
        y2 = int(50 + i * 2)
        cv2.rectangle(frame, (300, y2), (360, y2+80), (255, 0, 0), -1)
        cv2.putText(frame, "2", (320, y2+45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Object 3: Moving diagonally
        x3 = int(450 - i * 2)
        y3 = int(300 + i * 1)
        cv2.rectangle(frame, (x3, y3), (x3+50, y3+70), (0, 0, 255), -1)
        cv2.putText(frame, "3", (x3+15, y3+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add frame number
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    
    # Create annotations directory
    annotations_dir = os.path.join(dummy_dir, 'annotations')
    os.makedirs(annotations_dir, exist_ok=True)
    
    # Create dummy annotations file
    import json
    annotations = {
        'video': 'test_video.mp4',
        'num_frames': 100,
        'objects': [
            {'id': 1, 'class': 'person', 'action': 'walking'},
            {'id': 2, 'class': 'person', 'action': 'standing'},
            {'id': 3, 'class': 'person', 'action': 'running'}
        ]
    }
    
    annotations_path = os.path.join(annotations_dir, 'test_annotations.json')
    with open(annotations_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"✓ Dummy data created at: {dummy_dir}")
    print(f"  Test video: {video_path}")
    print(f"  Annotations: {annotations_path}")


def main():
    args = parse_args()
    
    # Create data directory
    os.makedirs(args.data_dir, exist_ok=True)
    print(f"Data directory: {args.data_dir}")
    
    # Download requested datasets
    if args.dataset == 'mot17' or args.dataset == 'all':
        download_mot17(args.data_dir)
    
    if args.dataset == 'mot20' or args.dataset == 'all':
        download_mot20(args.data_dir)
    
    if args.dataset == 'ava' or args.dataset == 'all':
        download_ava(args.data_dir)
    
    if args.dataset == 'dummy' or args.dataset == 'all':
        create_dummy_data(args.data_dir)
    
    print("\n" + "="*60)
    print("Dataset preparation complete!")
    print("="*60)


if __name__ == '__main__':
    main()