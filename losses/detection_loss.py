import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionLoss(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_obj=1.0, lambda_noobj=0.5, lambda_cls=1.0, iou_threshold=0.5):
        super(DetectionLoss, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_cls = lambda_cls
        self.iou_threshold = iou_threshold
    
    def forward(self, predictions, targets):
        pred_bbox = predictions['bbox']
        pred_obj = predictions['obj']
        pred_cls = predictions['cls']

        # Handling 5D grid output with multiple anchors
        if pred_bbox.dim() == 5:
            B, T, C, H, W = pred_bbox.shape
            # Reshape bbox: [B, T, anchors*4, H, W] -> [B, T*H*W*anchors, 4]
            pred_bbox = pred_bbox.permute(0, 1, 3, 4, 2).reshape(B, -1, 4)
            # Reshape obj: [B, T, anchors, H, W] -> [B, T*H*W*anchors]
            pred_obj = pred_obj.reshape(B, -1)
            # Reshape cls: [B, T, anchors*classes, H, W] -> [B, T*H*W*anchors, classes]
            if pred_cls is not None:
                num_classes = pred_cls.shape[2] // (C // 4)
                pred_cls = pred_cls.permute(0, 1, 3, 4, 2).reshape(B, -1, num_classes)

        gt_boxes = targets['boxes']
        gt_labels = targets['labels']
        obj_mask = targets.get('obj_mask', None)
        device = pred_bbox.device
        
        loss_bbox = torch.tensor(0.0, device=device)
        loss_obj = torch.tensor(0.0, device=device)
        loss_cls = torch.tensor(0.0, device=device)
        
        if obj_mask is None:
            obj_mask = self._create_object_mask(pred_bbox, gt_boxes)
        
        # 1. BBox Loss
        if obj_mask.sum() > 0:
            pos_pred_bbox = pred_bbox[obj_mask]
            matched_gt_boxes = self._match_targets(pos_pred_bbox, gt_boxes, obj_mask)
            loss_bbox = self.giou_loss(pos_pred_bbox, matched_gt_boxes).mean() * self.lambda_coord
        
        # 2. Objectness Loss
        if obj_mask.sum() > 0:
            loss_obj_pos = F.binary_cross_entropy_with_logits(pred_obj[obj_mask], torch.ones_like(pred_obj[obj_mask])) * self.lambda_obj
        else:
            loss_obj_pos = torch.tensor(0.0, device=device)
        
        neg_mask = ~obj_mask
        if neg_mask.sum() > 0:
            loss_obj_neg = F.binary_cross_entropy_with_logits(pred_obj[neg_mask], torch.zeros_like(pred_obj[neg_mask])) * self.lambda_noobj
        else:
            loss_obj_neg = torch.tensor(0.0, device=device)
        loss_obj = loss_obj_pos + loss_obj_neg
        
        # 3. Class Loss
        if obj_mask.sum() > 0 and pred_cls is not None:
            pos_pred_cls = pred_cls[obj_mask]
            matched_gt_labels = self._match_labels(obj_mask, gt_labels)
            loss_cls = F.cross_entropy(pos_pred_cls, matched_gt_labels) * self.lambda_cls
        
        return {'detection_loss': loss_bbox + loss_obj + loss_cls, 'bbox_loss': loss_bbox, 'obj_loss': loss_obj, 'cls_loss': loss_cls}

    def _create_object_mask(self, pred_boxes, gt_boxes):
        B, N, _ = pred_boxes.shape
        obj_mask = torch.zeros(B, N, dtype=torch.bool, device=pred_boxes.device)
        for b in range(B):
            current_gt = gt_boxes[b] if gt_boxes.dim() == 3 else gt_boxes
            num_gt = min(current_gt.shape[0], N)
            obj_mask[b, :num_gt] = True
        return obj_mask

    def giou_loss(self, pred_boxes, target_boxes):
        pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2
        target_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
        target_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
        target_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
        target_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        union_area = (pred_x2-pred_x1)*(pred_y2-pred_y1) + (target_x2-target_x1)*(target_y2-target_y1) - inter_area
        iou = inter_area / (union_area + 1e-7)
        ex1, ey1 = torch.min(pred_x1, target_x1), torch.min(pred_y1, target_y1)
        ex2, ey2 = torch.max(pred_x2, target_x2), torch.max(pred_y2, target_y2)
        enclose_area = (ex2-ex1)*(ey2-ey1)
        return 1 - (iou - (enclose_area - union_area) / (enclose_area + 1e-7))

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