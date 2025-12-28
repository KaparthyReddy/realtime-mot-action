import torch
import torch.optim as optim


def build_optimizer(model, config):
    """
    Build optimizer based on configuration
    
    Args:
        model: PyTorch model
        config: Dictionary with optimizer configuration
            - type: 'adam', 'adamw', 'sgd', 'rmsprop'
            - lr: Learning rate
            - weight_decay: L2 regularization
            - momentum: For SGD
            - betas: For Adam/AdamW
    
    Returns:
        PyTorch optimizer
    """
    optimizer_type = config.get('type', 'adamw').lower()
    lr = config.get('lr', 1e-4)
    weight_decay = config.get('weight_decay', 1e-4)
    
    # Separate parameters for different learning rates
    # Backbone typically needs lower learning rate
    param_groups = get_param_groups(model, lr, weight_decay)
    
    if optimizer_type == 'adam':
        betas = config.get('betas', (0.9, 0.999))
        optimizer = optim.Adam(
            param_groups,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay
        )
    
    elif optimizer_type == 'adamw':
        betas = config.get('betas', (0.9, 0.999))
        optimizer = optim.AdamW(
            param_groups,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay
        )
    
    elif optimizer_type == 'sgd':
        momentum = config.get('momentum', 0.9)
        nesterov = config.get('nesterov', True)
        optimizer = optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
    
    elif optimizer_type == 'rmsprop':
        momentum = config.get('momentum', 0.9)
        optimizer = optim.RMSprop(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer


def get_param_groups(model, base_lr, weight_decay):
    """
    Separate parameters into groups with different learning rates
    
    Typically:
    - Backbone: 0.1x base_lr
    - Heads: 1.0x base_lr
    - Batch norm: no weight decay
    """
    # Parameters that should not have weight decay
    no_decay_names = ['bias', 'bn', 'norm']
    
    # Separate backbone and other parameters
    backbone_params = []
    backbone_params_no_decay = []
    other_params = []
    other_params_no_decay = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check if parameter should have weight decay
        has_decay = not any(nd in name.lower() for nd in no_decay_names)
        
        # Check if parameter is in backbone
        is_backbone = 'backbone' in name.lower()
        
        if is_backbone:
            if has_decay:
                backbone_params.append(param)
            else:
                backbone_params_no_decay.append(param)
        else:
            if has_decay:
                other_params.append(param)
            else:
                other_params_no_decay.append(param)
    
    param_groups = [
        {
            'params': backbone_params,
            'lr': base_lr * 0.1,  # Lower LR for backbone
            'weight_decay': weight_decay
        },
        {
            'params': backbone_params_no_decay,
            'lr': base_lr * 0.1,
            'weight_decay': 0.0
        },
        {
            'params': other_params,
            'lr': base_lr,
            'weight_decay': weight_decay
        },
        {
            'params': other_params_no_decay,
            'lr': base_lr,
            'weight_decay': 0.0
        }
    ]
    
    # Filter out empty groups
    param_groups = [g for g in param_groups if len(g['params']) > 0]
    
    return param_groups


class SAM(torch.optim.Optimizer):
    """
    Sharpness Aware Minimization (SAM)
    
    Improves generalization by finding flat minima
    Reference: "Sharpness-Aware Minimization for Efficiently Improving Generalization"
    """
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """First step: compute and apply adversarial perturbation"""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # Store original parameter
                self.state[p]["old_p"] = p.data.clone()
                
                # Apply perturbation
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale
                p.add_(e_w)
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Second step: restore original parameters and update"""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # Restore original parameter
                p.data = self.state[p]["old_p"]
        
        # Update with base optimizer
        self.base_optimizer.step()
        
        if zero_grad:
            self.zero_grad()
    
    def _grad_norm(self):
        """Compute gradient norm"""
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm
    
    def step(self, closure=None):
        """Should not be called directly - use first_step and second_step"""
        raise NotImplementedError("Use first_step() and second_step() instead")


if __name__ == "__main__":
    print("Testing Optimizer Building:\n")
    
    # Create a dummy model
    import sys
    sys.path.append('..')
    from models.mot_action_net import MOTActionNet
    
    model = MOTActionNet(num_classes=5, num_actions=10)
    
    # Test different optimizers
    configs = [
        {'type': 'adam', 'lr': 1e-4, 'weight_decay': 1e-4},
        {'type': 'adamw', 'lr': 1e-4, 'weight_decay': 1e-4},
        {'type': 'sgd', 'lr': 1e-2, 'momentum': 0.9},
    ]
    
    for config in configs:
        optimizer = build_optimizer(model, config)
        print(f"Built {config['type'].upper()} optimizer:")
        print(f"  Number of param groups: {len(optimizer.param_groups)}")
        for i, group in enumerate(optimizer.param_groups):
            print(f"  Group {i}: {len(group['params'])} params, lr={group['lr']:.2e}")
        print()
    
    # Test SAM optimizer
    print("Testing SAM optimizer:")
    base_config = {'lr': 1e-4, 'weight_decay': 1e-4}
    sam_optimizer = SAM(model.parameters(), torch.optim.AdamW, rho=0.05, **base_config)
    print(f"  SAM optimizer created with {len(sam_optimizer.param_groups)} param groups")