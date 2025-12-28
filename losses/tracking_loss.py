import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class TrackingLoss(nn.Module):
    def __init__(self, margin=0.3, lambda_triplet=1.0, lambda_id=1.0):
        super(TrackingLoss, self).__init__()
        self.margin = margin
        self.lambda_triplet = lambda_triplet
        self.lambda_id = lambda_id
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)
    
    def forward(self, reid_features, track_ids):
        """
        Args:
            reid_features: (B, N_feat, D) 
            track_ids: (B, N_ids) 
        """
        device = reid_features.device
        B, N_feat, D = reid_features.shape
        _, N_ids = track_ids.shape
        
        # Safety check for dummy data or temporal mismatches
        if N_feat != N_ids:
            # If we have more IDs than features, truncate. If more features, truncate features.
            # This prevents the IndexError during training.
            min_n = min(N_feat, N_ids)
            reid_features = reid_features[:, :min_n, :]
            track_ids = track_ids[:, :min_n]
            N_feat = min_n

        loss_triplet = torch.tensor(0.0, device=device)
        loss_id = torch.tensor(0.0, device=device)
        num_triplets = 0
        
        for b in range(B):
            # track_ids[b] now guaranteed to match reid_features[b] dimension
            valid_mask = track_ids[b] >= 0
            if valid_mask.sum() < 2:
                continue
            
            valid_ids = track_ids[b][valid_mask]
            valid_features = reid_features[b][valid_mask]
            
            for i, anchor_id in enumerate(valid_ids):
                anchor_feat = valid_features[i:i+1]
                
                # Find positive (same ID, different index)
                pos_mask = (valid_ids == anchor_id) & (torch.arange(len(valid_ids), device=device) != i)
                if pos_mask.sum() == 0: continue
                
                # Find negative (different ID)
                neg_mask = valid_ids != anchor_id
                if neg_mask.sum() == 0: continue
                
                pos_idx = torch.nonzero(pos_mask)[0].item()
                neg_idx = torch.nonzero(neg_mask)[0].item()
                
                positive = valid_features[pos_idx:pos_idx+1]
                negative = valid_features[neg_idx:neg_idx+1]
                
                loss_triplet += self.triplet_loss(anchor_feat, positive, negative)
                num_triplets += 1
        
        if num_triplets > 0:
            loss_triplet = (loss_triplet / num_triplets) * self.lambda_triplet
        
        return {
            'tracking_loss': loss_triplet + loss_id,
            'triplet_loss': loss_triplet,
            'id_loss': loss_id
        }


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning ReID features.
    Fixed the indexing logic for calculating the probability of positive pairs.
    """
    def __init__(self, margin=1.0, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(self, features, labels):
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Compute similarity matrix [N, N]
        sim_matrix = torch.matmul(features, features.t()) / self.temperature
        
        # Create label matrix [N, N]
        labels = labels.unsqueeze(1)
        label_matrix = (labels == labels.t()).float()
        
        # Mask out diagonal (self-similarity)
        mask = torch.eye(len(features), device=features.device).bool()
        label_matrix = label_matrix.masked_fill(mask, 0)
        
        # Compute the denominator for all entries: sum(exp(sim)) along the rows
        exp_sim = torch.exp(sim_matrix)
        # Masking self-similarity in denominator is common in InfoNCE
        exp_sim = exp_sim.masked_fill(mask, 0) 
        sim_sum = exp_sim.sum(dim=1, keepdim=True) + 1e-8
        
        # Probability matrix [N, N]
        prob_matrix = exp_sim / sim_sum
        
        # Positive pairs mask
        pos_mask = label_matrix == 1
        
        if pos_mask.sum() > 0:
            # Take only the probabilities of the actual positive pairs
            pos_probs = prob_matrix[pos_mask]
            loss_pos = -torch.log(pos_probs + 1e-8).mean()
        else:
            loss_pos = torch.tensor(0.0, device=features.device)
        
        return loss_pos


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, lambda_c=0.001):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lambda_c = lambda_c
        
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)
    
    def forward(self, features, labels):
        centers_batch = self.centers[labels]
        loss = F.mse_loss(features, centers_batch)
        return loss * self.lambda_c


if __name__ == "__main__":
    print("Testing Tracking Loss:\n")
    track_loss = TrackingLoss(margin=0.3)
    
    B, N, D = 2, 50, 128
    reid_features = F.normalize(torch.randn(B, N, D), p=2, dim=-1)
    track_ids = torch.randint(-1, 10, (B, N))
    
    losses = track_loss(reid_features, track_ids)
    print("Tracking Losses:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")
    
    print("\nTesting Contrastive Loss:")
    contrastive_loss = ContrastiveLoss()
    features = F.normalize(torch.randn(100, 128), p=2, dim=1)
    labels = torch.randint(0, 10, (100,))
    cl = contrastive_loss(features, labels)
    print(f"  Contrastive loss: {cl.item():.4f}")
    
    print("\nTesting Center Loss:")
    center_loss = CenterLoss(num_classes=10, feat_dim=128)
    features = torch.randn(100, 128)
    labels = torch.randint(0, 10, (100,))
    cent_l = center_loss(features, labels)
    print(f"  Center loss: {cent_l.item():.4f}")