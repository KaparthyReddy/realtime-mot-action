import os
import torch
import shutil
from pathlib import Path


def save_checkpoint(state, is_best=False, checkpoint_dir='checkpoints', filename='checkpoint.pth'):
    """
    Save model checkpoint
    
    Args:
        state: Dictionary containing model state, optimizer state, etc.
        is_best: Whether this is the best model so far
        checkpoint_dir: Directory to save checkpoints
        filename: Checkpoint filename
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'best_model.pth')
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cpu'):
    """
    Load model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load checkpoint on
    
    Returns:
        Dictionary with epoch, best metrics, etc.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    info = {
        'epoch': checkpoint.get('epoch', 0),
        'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
        'global_step': checkpoint.get('global_step', 0)
    }
    
    print(f"✓ Checkpoint loaded successfully (epoch {info['epoch']})")
    
    return info


def get_latest_checkpoint(checkpoint_dir):
    """
    Get path to latest checkpoint
    
    Args:
        checkpoint_dir: Directory containing checkpoints
    
    Returns:
        Path to latest checkpoint or None
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    # Look for latest.pth first
    latest_path = checkpoint_dir / 'latest.pth'
    if latest_path.exists():
        return str(latest_path)
    
    # Otherwise find most recent checkpoint
    checkpoints = list(checkpoint_dir.glob('*.pth'))
    if not checkpoints:
        return None
    
    # Sort by modification time
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return str(latest)


def clean_old_checkpoints(checkpoint_dir, keep_best=True, keep_latest=True, keep_last_n=3):
    """
    Clean up old checkpoints to save disk space
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_best: Keep best_model.pth
        keep_latest: Keep latest.pth
        keep_last_n: Keep last N epoch checkpoints
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return
    
    # Files to keep
    keep_files = set()
    
    if keep_best:
        keep_files.add('best_model.pth')
    
    if keep_latest:
        keep_files.add('latest.pth')
    
    # Get all epoch checkpoints
    epoch_checkpoints = sorted(
        checkpoint_dir.glob('checkpoint_epoch_*.pth'),
        key=lambda p: int(p.stem.split('_')[-1])
    )
    
    # Keep last N
    if keep_last_n > 0:
        for checkpoint in epoch_checkpoints[-keep_last_n:]:
            keep_files.add(checkpoint.name)
    
    # Delete old checkpoints
    deleted_count = 0
    for checkpoint in checkpoint_dir.glob('*.pth'):
        if checkpoint.name not in keep_files:
            checkpoint.unlink()
            deleted_count += 1
    
    if deleted_count > 0:
        print(f"Cleaned up {deleted_count} old checkpoints")


def export_model_for_inference(model, export_path, input_shape=(1, 3, 640, 640)):
    """
    Export model for inference (scripted or traced)
    
    Args:
        model: PyTorch model
        export_path: Path to save exported model
        input_shape: Input shape for tracing
    """
    model.eval()
    
    # Try scripting first (more flexible)
    try:
        scripted_model = torch.jit.script(model)
        scripted_model.save(export_path)
        print(f"✓ Model scripted and saved to {export_path}")
    except Exception as e:
        print(f"Scripting failed: {e}")
        print("Trying tracing instead...")
        
        # Fall back to tracing
        try:
            dummy_input = torch.randn(input_shape)
            traced_model = torch.jit.trace(model, dummy_input)
            traced_model.save(export_path)
            print(f"✓ Model traced and saved to {export_path}")
        except Exception as e:
            print(f"Tracing also failed: {e}")
            print("Saving as regular checkpoint instead")
            torch.save(model.state_dict(), export_path)


if __name__ == "__main__":
    print("Testing Checkpoint Utilities:\n")
    
    import torch.nn as nn
    
    # Create dummy model
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Test saving
    print("1. Testing checkpoint saving...")
    checkpoint_dir = './test_checkpoints'
    
    state = {
        'epoch': 10,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': 0.5,
        'global_step': 1000
    }
    
    save_checkpoint(state, is_best=False, checkpoint_dir=checkpoint_dir)
    save_checkpoint(state, is_best=True, checkpoint_dir=checkpoint_dir)
    print("   ✓ Checkpoints saved")
    
    # Test loading
    print("\n2. Testing checkpoint loading...")
    new_model = nn.Linear(10, 5)
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
    
    info = load_checkpoint(
        os.path.join(checkpoint_dir, 'best_model.pth'),
        new_model,
        new_optimizer
    )
    print(f"   Loaded epoch: {info['epoch']}")
    print(f"   Best val loss: {info['best_val_loss']}")
    
    # Test getting latest
    print("\n3. Testing get latest checkpoint...")
    latest = get_latest_checkpoint(checkpoint_dir)
    print(f"   Latest checkpoint: {latest}")
    
    # Test cleanup
    print("\n4. Testing checkpoint cleanup...")
    # Create some dummy checkpoints
    for i in range(10):
        save_checkpoint(
            state,
            checkpoint_dir=checkpoint_dir,
            filename=f'checkpoint_epoch_{i}.pth'
        )
    
    print(f"   Created 10 checkpoints")
    clean_old_checkpoints(checkpoint_dir, keep_last_n=3)
    
    remaining = list(Path(checkpoint_dir).glob('*.pth'))
    print(f"   Remaining checkpoints: {len(remaining)}")
    
    # Cleanup test directory
    import shutil
    shutil.rmtree(checkpoint_dir)
    print("\n✓ All checkpoint tests passed!")