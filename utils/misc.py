import random
import numpy as np
import torch
import os


def set_seed(seed=42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make cudnn deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✓ Random seed set to {seed}")


def count_parameters(model, trainable_only=True):
    """
    Count model parameters
    
    Args:
        model: PyTorch model
        trainable_only: Only count trainable parameters
    
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_model_size(model, unit='MB'):
    """
    Get model size in memory
    
    Args:
        model: PyTorch model
        unit: 'MB' or 'GB'
    
    Returns:
        Model size
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    total_size = param_size + buffer_size
    
    if unit == 'MB':
        return total_size / 1024 / 1024
    elif unit == 'GB':
        return total_size / 1024 / 1024 / 1024
    else:
        return total_size


def print_model_summary(model):
    """
    Print model summary
    
    Args:
        model: PyTorch model
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    model_size = get_model_size(model, unit='MB')
    
    print("\n" + "="*60)
    print("Model Summary")
    print("="*60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Model size: {model_size:.2f} MB")
    print("="*60 + "\n")


def get_lr(optimizer):
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def freeze_layers(model, freeze_bn=True):
    """
    Freeze model layers (except batch norm by default)
    
    Args:
        model: PyTorch model
        freeze_bn: Whether to freeze batch norm layers
    """
    for name, param in model.named_parameters():
        if freeze_bn or 'bn' not in name.lower():
            param.requires_grad = False
    
    print(f"✓ Layers frozen (freeze_bn={freeze_bn})")


def unfreeze_layers(model):
    """Unfreeze all model layers"""
    for param in model.parameters():
        param.requires_grad = True
    
    print("✓ All layers unfrozen")


def format_time(seconds):
    """
    Format time in seconds to readable string
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def create_exp_dir(base_dir, exp_name=None):
    """
    Create experiment directory with timestamp
    
    Args:
        base_dir: Base directory for experiments
        exp_name: Experiment name (optional)
    
    Returns:
        Path to experiment directory
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if exp_name:
        exp_dir = os.path.join(base_dir, f"{exp_name}_{timestamp}")
    else:
        exp_dir = os.path.join(base_dir, timestamp)
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'logs'), exist_ok=True)
    
    print(f"✓ Experiment directory created: {exp_dir}")
    
    return exp_dir


if __name__ == "__main__":
    print("Testing Miscellaneous Utilities:\n")
    
    # Test seed setting
    print("1. Testing seed setting:")
    set_seed(42)
    print(f"   Random number: {random.random():.4f}")
    print(f"   Numpy random: {np.random.rand():.4f}")
    print(f"   Torch random: {torch.rand(1).item():.4f}")
    
    # Test with a model
    print("\n2. Testing model utilities:")
    import torch.nn as nn
    
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    print_model_summary(model)
    
    # Test freezing
    print("3. Testing layer freezing:")
    freeze_layers(model)
    trainable_before = count_parameters(model, trainable_only=True)
    print(f"   Trainable params after freezing: {trainable_before}")
    
    unfreeze_layers(model)
    trainable_after = count_parameters(model, trainable_only=True)
    print(f"   Trainable params after unfreezing: {trainable_after}")
    
    # Test time formatting
    print("\n4. Testing time formatting:")
    times = [30, 150, 7200]
    for t in times:
        print(f"   {t}s = {format_time(t)}")
    
    # Test experiment directory creation
    print("\n5. Testing experiment directory:")
    exp_dir = create_exp_dir('./test_experiments', 'test_run')
    
    # Cleanup
    import shutil
    shutil.rmtree('./test_experiments')
    
    print("\n✓ All miscellaneous utility tests passed!")