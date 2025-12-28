import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionLoss(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_obj=1.0, lambda_noobj=0.5, lambda_cls=1.0):
        super(DetectionLoss, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_cls = lambda_cls
    
    def forward(self, predictions, targets):
        pred_bbox = predictions['bbox']
        pred_obj = predictions['obj']
        pred_cls = predictions['cls']
        device = pred_bbox.device

        # Reshape logic for grid outputs
        if pred_bbox.dim() == 5:
            B, T, C, H, W = pred_bbox.shape
            pred_bbox = pred_bbox.permute(0, 1, 3, 4, 2).reshape(B, -1, 4)
            pred_obj = pred_obj.reshape(B, -1)
            if pred_cls is not None:
                num_classes = pred_cls.shape[2] // (C // 4)
                pred_cls = pred_cls.permute(0, 1, 3, 4, 2).reshape(B, -1, num_classes)

        obj_mask = targets.get('obj_mask')
        if obj_mask is None:
            obj_mask = self._create_object_mask(pred_bbox, targets['boxes'])
        
        loss_bbox = torch.tensor(0.0, device=device)
        loss_cls = torch.tensor(0.0, device=device)
        
        # 1. BBox Loss (GIoU)
        if obj_mask.sum() > 0:
            pos_pred = pred_bbox[obj_mask]
            pos_gt = self._match_targets(pos_pred, targets['boxes'], obj_mask)
            giou_losses = self.giou_loss(pos_pred, pos_gt)
            loss_bbox = giou_losses.mean() * self.lambda_coord
        
        # 2. Objectness Loss (BCE)
        # Separate positive and negative to apply different lambdas
        loss_obj_pos = torch.tensor(0.0, device=device)
        if obj_mask.sum() > 0:
            loss_obj_pos = F.binary_cross_entropy_with_logits(
                pred_obj[obj_mask], torch.ones_like(pred_obj[obj_mask])
            ) * self.lambda_obj
            
        loss_obj_neg = torch.tensor(0.0, device=device)
        if (~obj_mask).sum() > 0:
            loss_obj_neg = F.binary_cross_entropy_with_logits(
                pred_obj[~obj_mask], torch.zeros_like(pred_obj[~obj_mask])
            ) * self.lambda_noobj
        
        loss_obj = loss_obj_pos + loss_obj_neg
        
        # 3. Class Loss
        if obj_mask.sum() > 0 and pred_cls is not None:
            pos_cls = pred_cls[obj_mask]
            gt_cls = self._match_labels(obj_mask, targets['labels'])
            loss_cls = F.cross_entropy(pos_cls, gt_cls) * self.lambda_cls
        
        # Clamp detection loss to be strictly positive
        detection_total = loss_bbox + loss_obj + loss_cls
        return {
            'detection_loss': torch.clamp(detection_total, min=1e-6),
            'bbox_loss': loss_bbox,
            'obj_loss': loss_obj,
            'cls_loss': loss_cls
        }

    def giou_loss(self, pred, target):
        """Stable Generalized IoU loss: L = 1 - GIoU"""
        # Convert (xc, yc, w, h) to (x1, y1, x2, y2)
        p_x1, p_y1 = pred[:, 0] - pred[:, 2]/2, pred[:, 1] - pred[:, 3]/2
        p_x2, p_y2 = pred[:, 0] + pred[:, 2]/2, pred[:, 1] + pred[:, 3]/2
        t_x1, t_y1 = target[:, 0] - target[:, 2]/2, target[:, 1] - target[:, 3]/2
        t_x2, t_y2 = target[:, 0] + target[:, 2]/2, target[:, 1] + target[:, 3]/2
        
        # Intersection
        i_x1, i_y1 = torch.max(p_x1, t_x1), torch.max(p_y1, t_y1)
        i_x2, i_y2 = torch.min(p_x2, t_x2), torch.min(p_y2, t_y2)
        inter_area = torch.clamp(i_x2 - i_x1, min=0) * torch.clamp(i_y2 - i_y1, min=0)
        
        # Union
        p_area = torch.clamp(p_x2 - p_x1, min=0) * torch.clamp(p_y2 - p_y1, min=0)
        t_area = torch.clamp(t_x2 - t_x1, min=0) * torch.clamp(t_y2 - t_y1, min=0)
        union_area = p_area + t_area - inter_area + 1e-7
        
        iou = inter_area / union_area
        
        # Smallest Enclosing Box
        e_x1, e_y1 = torch.min(p_x1, t_x1), torch.min(p_y1, t_y1)
        e_x2, e_y2 = torch.max(p_x2, t_x2), torch.max(p_y2, t_y2)
        enclose_area = (e_x2 - e_x1) * (e_y2 - e_y1) + 1e-7
        
        giou = iou - (enclose_area - union_area) / enclose_area
        return 1.0 - giou

    def _create_object_mask(self, pred_boxes, gt_boxes):
        B, N, _ = pred_boxes.shape
        obj_mask = torch.zeros(B, N, dtype=torch.bool, device=pred_boxes.device)
        for b in range(B):
            current_gt = gt_boxes[b] if gt_boxes.dim() == 3 else gt_boxes
            num_gt = min(current_gt.shape[0], N)
            obj_mask[b, :num_gt] = True
        return obj_mask

    def _match_targets(self, pos_pred_bbox, gt_boxes, obj_mask):
        matched = []
        indices = torch.nonzero(obj_mask)
        for idx in indices:
            b = idx[0].item()
            gt = gt_boxes[b] if gt_boxes.dim() == 3 else gt_boxes
            matched.append(gt[min(idx[1].item(), gt.shape[0]-1)])
        return torch.stack(matched)

    def _match_labels(self, obj_mask, gt_labels):
        matched = []
        indices = torch.nonzero(obj_mask)
        for idx in indices:
            b = idx[0].item()
            gt = gt_labels[b] if gt_labels.dim() == 2 else gt_labels
            matched.append(gt[min(idx[1].item(), gt.shape[0]-1)])
        return torch.stack(matched)