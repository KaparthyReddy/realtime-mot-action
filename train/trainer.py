import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from train.optimizer import build_optimizer
from train.scheduler import build_scheduler
from losses.combined_loss import CombinedLoss
from utils.logging_utils import AverageMeter, ProgressMeter
from utils.checkpoint import save_checkpoint, load_checkpoint


class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device='cuda', log_dir='./logs'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.log_dir = log_dir
        
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'checkpoints'), exist_ok=True)
        
        self.criterion = CombinedLoss(
            num_classes=config['num_classes'],
            num_actions=config['num_actions'],
            lambda_detection=config.get('lambda_detection', 1.0),
            lambda_tracking=config.get('lambda_tracking', 0.5),
            lambda_action=config.get('lambda_action', 1.0),
            use_uncertainty_weighting=config.get('use_uncertainty_weighting', False)
        ).to(device)
        
        self.optimizer = build_optimizer(model, config['optimizer'])
        self.scheduler = build_scheduler(self.optimizer, config['scheduler'], num_epochs=config['num_epochs'])
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard'))
        
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.global_step = 0
        self.max_grad_norm = config.get('max_grad_norm', 10.0)
        self.use_amp = config.get('use_amp', False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.log_interval = config.get('log_interval', 10)
        self.val_interval = config.get('val_interval', 1)
        
        print(f"Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Training samples: {len(train_loader.dataset)}")
        print(f"  Validation samples: {len(val_loader.dataset)}")
        print(f"  Batch size: {train_loader.batch_size}")
        print(f"  Number of epochs: {config['num_epochs']}")
    
    def train_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter('Loss', ':.4f')
        det_losses = AverageMeter('Det', ':.4f')
        track_losses = AverageMeter('Track', ':.4f')
        action_losses = AverageMeter('Action', ':.4f')
        batch_time = AverageMeter('Time', ':6.3f')
        
        progress = ProgressMeter(len(self.train_loader), [batch_time, losses, det_losses, track_losses, action_losses], prefix=f"Epoch: [{epoch+1}]")
        
        end = time.time()
        for i, batch in enumerate(self.train_loader):
            inputs = batch['frames'].to(self.device)
            targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items() if k != 'frames'}
            
            predictions = self.model(inputs, mode='all')
            loss_dict = self.criterion(predictions, targets)
            loss = loss_dict['total_loss']
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            losses.update(loss.item(), inputs.size(0))
            if 'detection_loss' in loss_dict:
                det_losses.update(loss_dict['detection_loss'].item(), inputs.size(0))
            if 'tracking_loss' in loss_dict:
                track_losses.update(loss_dict['tracking_loss'].item(), inputs.size(0))
            if 'action_loss' in loss_dict:
                action_losses.update(loss_dict['action_loss'].item(), inputs.size(0))
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % self.log_interval == 0:
                progress.display(i)
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
            
            self.global_step += 1
        
        return {'loss': losses.avg, 'det_loss': det_losses.avg, 'track_loss': track_losses.avg, 'action_loss': action_losses.avg}
    
    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        losses = AverageMeter('Loss', ':.4f')
        det_losses = AverageMeter('Det', ':.4f')
        track_losses = AverageMeter('Track', ':.4f')
        action_losses = AverageMeter('Action', ':.4f')
        action_acc = AverageMeter('ActionAcc', ':6.2f')
        
        for batch in tqdm(self.val_loader, desc='Validation', leave=False):
            inputs = batch['frames'].to(self.device)
            targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items() if k != 'frames'}
            
            predictions = self.model(inputs, mode='all')
            loss_dict = self.criterion(predictions, targets)
            loss = loss_dict['total_loss']
            
            losses.update(loss.item(), inputs.size(0))
            if 'detection_loss' in loss_dict:
                det_losses.update(loss_dict['detection_loss'].item(), inputs.size(0))
            if 'tracking_loss' in loss_dict:
                track_losses.update(loss_dict['tracking_loss'].item(), inputs.size(0))
            if 'action_loss' in loss_dict:
                action_losses.update(loss_dict['action_loss'].item(), inputs.size(0))
            if 'action_accuracy' in loss_dict:
                action_acc.update(loss_dict['action_accuracy'].item() * 100, inputs.size(0))
        
        print(f"\n{'='*80}")
        print(f"Validation Results - Epoch {epoch + 1}")
        print(f"{'='*80}")
        print(f"  Total Loss:     {losses.avg:.4f}")
        print(f"  Detection Loss: {det_losses.avg:.4f}")
        print(f"  Tracking Loss:  {track_losses.avg:.4f}")
        print(f"  Action Loss:    {action_losses.avg:.4f}")
        print(f"  Action Acc:     {action_acc.avg:.2f}%")
        print(f"{'='*80}\n")
        
        self.writer.add_scalar('val/loss', losses.avg, epoch)
        
        return {'loss': losses.avg, 'det_loss': det_losses.avg, 'track_loss': track_losses.avg, 'action_loss': action_losses.avg, 'action_acc': action_acc.avg}
    
    def train(self):
        print("\n" + "="*80)
        print("Starting training...")
        print("="*80 + "\n")
        
        best_epoch = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.start_epoch, self.config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
            print("-" * 80)
            
            train_metrics = self.train_epoch(epoch)
            train_losses.append(train_metrics['loss'])
            
            if hasattr(self.scheduler, 'step'):
                self.scheduler.step()
            
            if (epoch + 1) % self.val_interval == 0:
                val_metrics = self.validate(epoch)
                val_losses.append(val_metrics['loss'])
                
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    best_epoch = epoch + 1
                    self.save_checkpoint(epoch, is_best=True)
                    print(f"\n✓ Best model saved! Val loss: {val_metrics['loss']:.4f}")
            
            if (epoch + 1) % self.config.get('checkpoint_interval', 10) == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        print(f"Total epochs: {self.config['num_epochs']}")
        print(f"Best epoch: {best_epoch}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        if train_losses:
            print(f"Final training loss: {train_losses[-1]:.4f}")
            print(f"Initial training loss: {train_losses[0]:.4f}")
            print(f"Loss improvement: {train_losses[0] - train_losses[-1]:.4f}")
        print(f"\nCheckpoints saved to: {os.path.join(self.log_dir, 'checkpoints')}")
        print("="*80 + "\n")
        
        self.writer.close()
    
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self.scheduler, 'state_dict') else None,
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(self.log_dir, 'checkpoints', 'latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.log_dir, 'checkpoints', 'best.pth')
            torch.save(checkpoint, best_path)
        
        if (epoch + 1) % self.config.get('checkpoint_interval', 10) == 0:
            periodic_path = os.path.join(self.log_dir, 'checkpoints', f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, periodic_path)
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and hasattr(self.scheduler, 'load_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.global_step = checkpoint['global_step']