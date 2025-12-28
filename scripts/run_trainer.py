import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mot_action_net import MOTActionNet, MOTActionNetLite
from train.trainer import Trainer
from utils.misc import set_seed, print_model_summary, create_exp_dir


class DummyDataset(Dataset):
    def __init__(self, num_samples, temporal_window, image_size, num_classes, num_actions):
        self.num_samples = num_samples
        self.temporal_window = temporal_window
        self.image_size = image_size
        self.num_classes = num_classes
        self.num_actions = num_actions
        self.num_total_predictions = temporal_window * 20 * 20 * 3
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        num_gt_objects = self.temporal_window
        
        return {
            'frames': torch.randn(self.temporal_window, 3, self.image_size, self.image_size),
            'boxes': torch.rand(num_gt_objects, 4) * self.image_size,
            'labels': torch.randint(0, self.num_classes, (num_gt_objects,)),
            'track_ids': torch.randint(0, 20, (num_gt_objects,)),
            'obj_mask': torch.cat([
                torch.ones(num_gt_objects, dtype=torch.bool),
                torch.zeros(self.num_total_predictions - num_gt_objects, dtype=torch.bool)
            ]),
            'actions': torch.randint(0, self.num_actions, (1,))[0]
        }


def parse_args():
    parser = argparse.ArgumentParser(description='Train MOTActionNet')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='full', choices=['full', 'lite'])
    parser.add_argument('--num-classes', type=int, default=1)
    parser.add_argument('--num-actions', type=int, default=10)
    parser.add_argument('--temporal-window', type=int, default=8)
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    
    # Loss weights
    parser.add_argument('--lambda-detection', type=float, default=1.0)
    parser.add_argument('--lambda-tracking', type=float, default=0.5)
    parser.add_argument('--lambda-action', type=float, default=1.0)
    parser.add_argument('--use-uncertainty-weighting', action='store_true')
    
    # Optimizer and scheduler
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['step', 'multistep', 'cosine', 'plateau'])
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='dummy')
    
    # Checkpointing
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--checkpoint-interval', type=int, default=10)
    
    # Logging
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--log-interval', type=int, default=10)
    
    # Hardware
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--use-amp', action='store_true')
    
    # Miscellaneous
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', type=str, default=None)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create experiment directory
    exp_dir = create_exp_dir(args.log_dir, args.exp_name)
    
    # Set device
    device = args.device
    print(f"\nUsing device: {device}")
    
    # Create model
    print("\nCreating model...")
    if args.model == 'full':
        model = MOTActionNet(
            num_classes=args.num_classes,
            num_actions=args.num_actions,
            temporal_window=args.temporal_window
        )
    else:
        model = MOTActionNetLite(
            num_classes=args.num_classes,
            num_actions=args.num_actions,
            temporal_window=args.temporal_window
        )
    
    print_model_summary(model)
    
    # Create datasets
    print("\nLoading datasets...")
    if args.dataset == 'dummy':
        print("Using dummy dataset for testing")
        train_dataset = DummyDataset(100, args.temporal_window, 320, args.num_classes, args.num_actions)
        val_dataset = DummyDataset(20, args.temporal_window, 320, args.num_classes, args.num_actions)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented yet")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Training configuration
    training_config = {
        'num_classes': args.num_classes,
        'num_actions': args.num_actions,
        'num_epochs': args.num_epochs,
        'optimizer': {
            'type': args.optimizer,
            'lr': args.lr,
            'weight_decay': args.weight_decay
        },
        'scheduler': {
            'type': args.scheduler,
            'T_max': args.num_epochs,
            'eta_min': 1e-6
        },
        'lambda_detection': args.lambda_detection,
        'lambda_tracking': args.lambda_tracking,
        'lambda_action': args.lambda_action,
        'use_uncertainty_weighting': args.use_uncertainty_weighting,
        'max_grad_norm': 10.0,
        'log_interval': args.log_interval,
        'val_interval': 1,
        'checkpoint_interval': args.checkpoint_interval,
        'use_amp': args.use_amp
    }
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device,
        log_dir=exp_dir
    )
    
    # Resume from checkpoint if provided
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint(trainer.start_epoch - 1, is_best=False)
        print("✓ Checkpoint saved")
    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()