import torch
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR, 
    CosineAnnealingLR, ReduceLROnPlateau,
    OneCycleLR, CosineAnnealingWarmRestarts
)
import math


def build_scheduler(optimizer, config, num_epochs=None):
    """
    Build learning rate scheduler
    
    Args:
        optimizer: PyTorch optimizer
        config: Dictionary with scheduler configuration
            - type: Scheduler type
            - Various scheduler-specific parameters
        num_epochs: Total number of epochs (for some schedulers)
    
    Returns:
        PyTorch scheduler
    """
    scheduler_type = config.get('type', 'cosine').lower()
    
    if scheduler_type == 'step':
        step_size = config.get('step_size', 30)
        gamma = config.get('gamma', 0.1)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_type == 'multistep':
        milestones = config.get('milestones', [30, 60, 90])
        gamma = config.get('gamma', 0.1)
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
    elif scheduler_type == 'exponential':
        gamma = config.get('gamma', 0.95)
        scheduler = ExponentialLR(optimizer, gamma=gamma)
    
    elif scheduler_type == 'cosine':
        T_max = config.get('T_max', num_epochs if num_epochs else 100)
        eta_min = config.get('eta_min', 1e-6)
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    
    elif scheduler_type == 'cosine_warm_restarts':
        T_0 = config.get('T_0', 10)
        T_mult = config.get('T_mult', 2)
        eta_min = config.get('eta_min', 1e-6)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )
    
    elif scheduler_type == 'plateau':
        mode = config.get('mode', 'min')
        factor = config.get('factor', 0.1)
        patience = config.get('patience', 10)
        scheduler = ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience
        )
    
    elif scheduler_type == 'onecycle':
        max_lr = config.get('max_lr', 1e-3)
        epochs = config.get('epochs', num_epochs if num_epochs else 100)
        steps_per_epoch = config.get('steps_per_epoch', 100)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch
        )
    
    elif scheduler_type == 'warmup_cosine':
        warmup_epochs = config.get('warmup_epochs', 5)
        total_epochs = num_epochs if num_epochs else config.get('total_epochs', 100)
        eta_min = config.get('eta_min', 1e-6)
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            eta_min=eta_min
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler


class WarmupCosineScheduler:
    """
    Cosine annealing scheduler with linear warmup
    
    Gradually increases learning rate during warmup,
    then applies cosine annealing
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        
        # Store initial learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0
    
    def step(self, epoch=None):
        """Update learning rate"""
        if epoch is None:
            epoch = self.current_epoch + 1
        
        self.current_epoch = epoch
        
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr_scale = (epoch + 1) / self.warmup_epochs
            lrs = [base_lr * lr_scale for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr_scale = self.eta_min + (1 - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
            lrs = [base_lr * lr_scale for base_lr in self.base_lrs]
        
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr
    
    def get_last_lr(self):
        """Get current learning rates"""
        return [group['lr'] for group in self.optimizer.param_groups]


class PolynomialLRScheduler:
    """
    Polynomial learning rate decay
    
    LR decays as (1 - epoch/total_epochs)^power
    """
    def __init__(self, optimizer, total_epochs, power=0.9, min_lr=1e-6):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.power = power
        self.min_lr = min_lr
        
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0
    
    def step(self, epoch=None):
        """Update learning rate"""
        if epoch is None:
            epoch = self.current_epoch + 1
        
        self.current_epoch = epoch
        
        factor = (1 - epoch / self.total_epochs) ** self.power
        lrs = [max(base_lr * factor, self.min_lr) for base_lr in self.base_lrs]
        
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr
    
    def get_last_lr(self):
        """Get current learning rates"""
        return [group['lr'] for group in self.optimizer.param_groups]


if __name__ == "__main__":
    print("Testing Learning Rate Schedulers:\n")
    
    # Create dummy optimizer
    import torch.nn as nn
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Test different schedulers
    configs = [
        {'type': 'step', 'step_size': 10, 'gamma': 0.5},
        {'type': 'multistep', 'milestones': [10, 20, 30], 'gamma': 0.1},
        {'type': 'cosine', 'T_max': 50, 'eta_min': 1e-6},
        {'type': 'warmup_cosine', 'warmup_epochs': 5, 'total_epochs': 50},
    ]
    
    for config in configs:
        print(f"Testing {config['type']} scheduler:")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = build_scheduler(optimizer, config, num_epochs=50)
        
        lrs = []
        for epoch in range(50):
            scheduler.step()
            lrs.append(optimizer.param_groups[0]['lr'])
        
        print(f"  LR at epoch 0: {lrs[0]:.2e}")
        print(f"  LR at epoch 10: {lrs[10]:.2e}")
        print(f"  LR at epoch 25: {lrs[25]:.2e}")
        print(f"  LR at epoch 49: {lrs[49]:.2e}")
        print()