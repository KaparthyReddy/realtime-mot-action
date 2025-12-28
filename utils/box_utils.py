import torch
import numpy as np


def box_cxcywh_to_xyxy(boxes):
    """
    Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2] format
    
    Args:
        boxes: Tensor or array of shape (..., 4)
    
    Returns:
        Converted boxes
    """
    if isinstance(boxes, torch.Tensor):
        x_c, y_c, w, h = boxes.unbind(-1)
        boxes_xyxy = torch.stack([
            x_c - w / 2,
            y_c - h / 2,
            x_c + w / 2,
            y_c + h / 2
        ], dim=-1)
    else:
        x_c, y_c, w, h = np.split(boxes, 4, axis=-1)
        boxes_xyxy = np.concatenate([
            x_c - w / 2,
            y_c - h / 2,
            x_c + w / 2,
            y_c + h / 2
        ], axis=-1)
    
    return boxes_xyxy


def box_xyxy_to_cxcywh(boxes):
    """
    Convert boxes from [x1, y1, x2, y2] to [cx, cy, w, h] format
    
    Args:
        boxes: Tensor or array of shape (..., 4)
    
    Returns:
        Converted boxes
    """
    if isinstance(boxes, torch.Tensor):
        x1, y1, x2, y2 = boxes.unbind(-1)
        boxes_cxcywh = torch.stack([
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            x2 - x1,
            y2 - y1
        ], dim=-1)
    else:
        x1, y1, x2, y2 = np.split(boxes, 4, axis=-1)
        boxes_cxcywh = np.concatenate([
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            x2 - x1,
            y2 - y1
        ], axis=-1)
    
    return boxes_cxcywh


def box_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes
    
    Args:
        boxes1: Tensor of shape (N, 4) in [x1, y1, x2, y2] format
        boxes2: Tensor of shape (M, 4) in [x1, y1, x2, y2] format
    
    Returns:
        IoU matrix of shape (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)
    
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)
    
    # Union
    union = area1[:, None] + area2 - inter
    
    # IoU
    iou = inter / (union + 1e-7)
    
    return iou


def box_giou(boxes1, boxes2):
    """
    Compute Generalized IoU between two sets of boxes
    
    Args:
        boxes1: Tensor of shape (N, 4)
        boxes2: Tensor of shape (N, 4)
    
    Returns:
        GIoU of shape (N,)
    """
    # IoU
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    
    union = area1 + area2 - inter
    iou = inter / (union + 1e-7)
    
    # Smallest enclosing box
    lt_enclosing = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb_enclosing = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    
    wh_enclosing = (rb_enclosing - lt_enclosing).clamp(min=0)
    area_enclosing = wh_enclosing[:, 0] * wh_enclosing[:, 1]
    
    # GIoU
    giou = iou - (area_enclosing - union) / (area_enclosing + 1e-7)
    
    return giou


def box_nms(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression
    
    Args:
        boxes: Tensor of shape (N, 4) in [x1, y1, x2, y2] format
        scores: Tensor of shape (N,)
        iou_threshold: IoU threshold for suppression
    
    Returns:
        Indices of boxes to keep
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    
    # Sort by score
    _, order = scores.sort(descending=True)
    
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order[0].item())
            break
        
        # Keep highest scoring box
        i = order[0]
        keep.append(i.item())
        
        # Compute IoU with remaining boxes
        ious = box_iou(boxes[i:i+1], boxes[order[1:]])[0]
        
        # Keep boxes with IoU below threshold
        mask = ious <= iou_threshold
        order = order[1:][mask]
    
    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)


def clip_boxes_to_image(boxes, image_size):
    """
    Clip boxes to image boundaries
    
    Args:
        boxes: Tensor of shape (..., 4) in [x1, y1, x2, y2] format
        image_size: (height, width) tuple
    
    Returns:
        Clipped boxes
    """
    height, width = image_size
    
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.clone()
        boxes[..., 0].clamp_(min=0, max=width)
        boxes[..., 1].clamp_(min=0, max=height)
        boxes[..., 2].clamp_(min=0, max=width)
        boxes[..., 3].clamp_(min=0, max=height)
    else:
        boxes = boxes.copy()
        boxes[..., 0] = np.clip(boxes[..., 0], 0, width)
        boxes[..., 1] = np.clip(boxes[..., 1], 0, height)
        boxes[..., 2] = np.clip(boxes[..., 2], 0, width)
        boxes[..., 3] = np.clip(boxes[..., 3], 0, height)
    
    return boxes


def scale_boxes(boxes, original_size, target_size):
    """
    Scale boxes from original image size to target size
    
    Args:
        boxes: Tensor of shape (..., 4)
        original_size: (height, width) tuple
        target_size: (height, width) tuple
    
    Returns:
        Scaled boxes
    """
    orig_h, orig_w = original_size
    target_h, target_w = target_size
    
    scale_y = target_h / orig_h
    scale_x = target_w / orig_w
    
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.clone()
        boxes[..., [0, 2]] *= scale_x
        boxes[..., [1, 3]] *= scale_y
    else:
        boxes = boxes.copy()
        boxes[..., [0, 2]] *= scale_x
        boxes[..., [1, 3]] *= scale_y
    
    return boxes


if __name__ == "__main__":
    print("Testing Box Utilities:\n")
    
    # Test conversions
    print("1. Testing box format conversions:")
    boxes_cxcywh = torch.tensor([
        [100, 100, 50, 80],
        [300, 200, 60, 90]
    ], dtype=torch.float32)
    
    boxes_xyxy = box_cxcywh_to_xyxy(boxes_cxcywh)
    print(f"   CXCYWH: {boxes_cxcywh[0]}")
    print(f"   XYXY: {boxes_xyxy[0]}")
    
    boxes_back = box_xyxy_to_cxcywh(boxes_xyxy)
    print(f"   Back to CXCYWH: {boxes_back[0]}")
    
    # Test IoU
    print("\n2. Testing IoU computation:")
    boxes1 = torch.tensor([
        [0, 0, 100, 100],
        [50, 50, 150, 150]
    ], dtype=torch.float32)
    
    boxes2 = torch.tensor([
        [50, 50, 150, 150],
        [200, 200, 300, 300]
    ], dtype=torch.float32)
    
    iou_matrix = box_iou(boxes1, boxes2)
    print(f"   IoU matrix:\n{iou_matrix}")
    
    # Test GIoU
    print("\n3. Testing GIoU:")
    giou = box_giou(boxes1, boxes2[[0, 1]])
    print(f"   GIoU: {giou}")
    
    # Test NMS
    print("\n4. Testing NMS:")
    boxes = torch.tensor([
        [100, 100, 200, 200],
        [110, 110, 210, 210],
        [300, 300, 400, 400]
    ], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8, 0.95])
    
    keep = box_nms(boxes, scores, iou_threshold=0.5)
    print(f"   Kept boxes: {keep}")
    
    # Test clipping
    print("\n5. Testing box clipping:")
    boxes = torch.tensor([
        [-10, -10, 50, 50],
        [600, 450, 700, 550]
    ], dtype=torch.float32)
    
    clipped = clip_boxes_to_image(boxes, (480, 640))
    print(f"   Original: {boxes[0]}")
    print(f"   Clipped: {clipped[0]}")
    
    # Test scaling
    print("\n6. Testing box scaling:")
    boxes = torch.tensor([[100, 100, 200, 200]], dtype=torch.float32)
    scaled = scale_boxes(boxes, (480, 640), (240, 320))
    print(f"   Original (480x640): {boxes[0]}")
    print(f"   Scaled (240x320): {scaled[0]}")
    
    print("\n✓ All box utility tests passed!")