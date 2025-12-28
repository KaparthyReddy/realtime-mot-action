import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mot_action_net import MOTActionNet, MOTActionNetLite
from evaluation.mot_metrics import MOTMetrics
from evaluation.action_metrics import ActionMetrics
from utils.checkpoint import load_checkpoint
from utils.misc import set_seed, print_model_summary


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate MOTActionNet')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--dataset', type=str, default='dummy',
                        help='Dataset name')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save predictions for visualization')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def create_dummy_dataset(num_samples=50):
    """Create dummy test dataset"""
    from torch.utils.data import Dataset
    
    class DummyTestDataset(Dataset):
        def __len__(self):
            return num_samples
        
        def __getitem__(self, idx):
            return {
                'frames': torch.randn(8, 3, 320, 320),
                'boxes': torch.rand(10, 4) * 320,
                'labels': torch.randint(0, 5, (10,)),
                'track_ids': torch.arange(10),
                'actions': torch.randint(0, 10, (1,))[0],
                'frame_id': idx
            }
    
    return DummyTestDataset()


@torch.no_grad()
def evaluate(model, data_loader, device):
    """Evaluate model on dataset"""
    model.eval()
    
    # Initialize metrics
    mot_metrics = MOTMetrics(iou_threshold=0.5)
    action_metrics = ActionMetrics(num_classes=10)
    
    # Storage for predictions
    all_predictions = []
    
    print("\nEvaluating...")
    for batch in tqdm(data_loader, desc='Evaluation'):
        # Move to device
        inputs = batch['frames'].to(device)
        
        # Forward pass
        predictions = model(inputs, mode='all')
        
        # Extract predictions
        # Detection predictions
        det_preds = predictions['detection']
        
        # Action predictions
        if predictions['action'] is not None:
            action_logits = predictions['action']
            action_preds = torch.argmax(action_logits, dim=1)
            action_targets = batch['actions'].to(device)
            
            action_metrics.update(
                action_preds.cpu().numpy(),
                action_targets.cpu().numpy(),
                torch.softmax(action_logits, dim=1).cpu().numpy()
            )
        
        # TODO: Process detection predictions for MOT metrics
        # This requires decoding the predictions and matching with ground truth
        
        # Store predictions
        all_predictions.append({
            'detection': det_preds,
            'action': action_preds.cpu() if predictions['action'] is not None else None
        })
    
    # Compute final metrics
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    
    # Action metrics
    action_results = action_metrics.compute()
    print("\nAction Recognition Metrics:")
    print(f"  Accuracy: {action_results['accuracy']:.2f}%")
    if 'top5_accuracy' in action_results:
        print(f"  Top-5 Accuracy: {action_results['top5_accuracy']:.2f}%")
    print(f"  Precision: {action_results['precision_macro']:.2f}%")
    print(f"  Recall: {action_results['recall_macro']:.2f}%")
    print(f"  F1-Score: {action_results['f1_macro']:.2f}%")
    
    # MOT metrics (if available)
    # mot_results = mot_metrics.compute()
    # print("\nMulti-Object Tracking Metrics:")
    # print(f"  MOTA: {mot_results['MOTA']:.2f}%")
    # print(f"  MOTP: {mot_results['MOTP']:.2f}%")
    # print(f"  IDF1: {mot_results['IDF1']:.2f}%")
    
    print("="*60)
    
    return {
        'action': action_results,
        # 'mot': mot_results,
        'predictions': all_predictions
    }


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading model...")
    # Determine model type from checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint.get('config', {})
    
    # Create model (adjust based on your checkpoint structure)
    model = MOTActionNet(
        num_classes=config.get('num_classes', 1),
        num_actions=config.get('num_actions', 10),
        temporal_window=config.get('temporal_window', 8)
    )
    
    # Load weights
    load_checkpoint(args.checkpoint, model, device=device)
    model = model.to(device)
    
    print_model_summary(model)
    
    # Load test dataset
    print("\nLoading test dataset...")
    if args.dataset == 'dummy':
        test_dataset = create_dummy_dataset(num_samples=50)
    else:
        # TODO: Load real test dataset
        raise NotImplementedError(f"Dataset {args.dataset} not implemented yet")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device == 'cuda' else False
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Evaluate
    results = evaluate(model, test_loader, device)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    import json
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if key != 'predictions':
                if isinstance(value, dict):
                    json_results[key] = {k: float(v) if hasattr(v, 'item') else v 
                                        for k, v in value.items() 
                                        if k != 'confusion_matrix' and k != 'per_class'}
        json.dump(json_results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_path}")
    
    # Save predictions if requested
    if args.save_predictions:
        preds_path = os.path.join(args.output_dir, 'predictions.pth')
        torch.save(results['predictions'], preds_path)
        print(f"✓ Predictions saved to {preds_path}")


if __name__ == '__main__':
    main()