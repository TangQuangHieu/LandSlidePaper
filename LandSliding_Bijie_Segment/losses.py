import torch
import torch.nn as nn
import torch.nn.functional as F

class IOULoss(nn.Module):
    def __init__(self, num_classes=2):
        super(IOULoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, y_true, y_pred):
        """
        # Calculate IOU loss on batch for multi-class segmentation.
        # Input:
            - y_true (3D torch tensor): BxWxH - ground truth labels (integer values)
            - y_pred (4D torch tensor): BxCxWxH - predicted class probabilities (already apply softmax)
        # Output:
            - iou loss: a scalar
        """
        ## prevent division by zero
        epsilon = 1e-7

        ## one-hot
        
        y_true = F.one_hot(y_true.long(), num_classes=self.num_classes) # BxWxHxC
        # print(y_true.shape)
        # y_true = y_true.permute(0, 2, 3, 1) # BxWxHxC

        ## reshape
        # y_true = y_true.view(-1, self.num_classes) # (B*W*H,C)
        y_pred = y_pred.permute(0,2,3,1) # BxWxHxC
        # print("y_pred.shape",y_pred.shape)
        # y_pred = y_pred.view(-1, self.num_classes) # (B*W*H,C)

        ## calculate iou loss of each class
        intersection = torch.sum(y_true * y_pred, dim=(0,1,2))               # |A| n |B|
        union = torch.sum(y_true + y_pred, dim=(0,1,2)) - intersection       # |A| u |B| - (|A|n|B|)

        iou = (intersection + epsilon) / (union + epsilon)
        # print(iou)
        ## calculate the IOU loss
        iou_loss = -torch.log(iou) # can use -log(iou) or 1 - iou
        # print(iou_loss.shape,torch.sum(iou_loss),torch.mean(iou_loss))
        b,w,h,c = y_true.shape
        return iou_loss.sum()/(b*w*h*c)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2., scale=1., num_classes=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.scale = scale
        self.num_classes = num_classes

    def forward(self, y_true, y_pred):
        """
        # Compute loss on batch
        # Input:
            - y_true (3D torch tensor): BxWxH - ground truth labels (integer values)
            - y_pred (4D torch tensor): BxWxHxC - predicted class probabilities (already apply softmax)
        # Output:
            - focal loss: a scalar
        """
        ## one-hot
        y_true = F.one_hot(y_true.long(), num_classes=self.num_classes) # BxWxHxC
        # y_true = y_true.view(-1,self.num_classes) #BxWxH,C
        ## avoid absolute zero and absolute one
        y_pred = torch.clamp(y_pred, 1e-7, 1. - 1e-7)
        y_pred = y_pred.permute(0,2,3,1)
        # y_pred = y_pred.view(-1,self.num_classes) #BxWxH,C
        # print("y_pred.shape",y_pred.shape,'y_true.shape',y_true.shape)
        ## cross entropy loss
        ce_loss = -y_true * torch.log(y_pred) # BxWxHxC

        ## downgrade background class (represented by 0) by scale times (0 <= scale <= 1)
        alpha = torch.ones_like(y_pred[..., :1]) * self.scale # BxWxHx1
        alpha = torch.cat([alpha, torch.ones_like(y_pred[..., :1])], dim=-1) # BxWxHx2

        ## focal weight
        
        focal_weight = torch.where(y_true == 1, 1 - y_pred, y_pred)
        # print('alpha.shape',alpha.shape,'focal_weight.shape',focal_weight.shape)
        focal_weight = alpha * (focal_weight ** self.gamma)

        ## focal loss
        focal_loss = torch.sum(focal_weight * ce_loss,dim=(0,1,2))
        # print(focal_loss.shape,torch.sum(focal_loss),torch.mean(focal_loss))
        b,w,h,c = y_true.shape
        return focal_loss.sum()/(b*w*h*c)

class PixelWiseFocalLoss(nn.Module):
    """
        # Example usage:
        # Assume `predictions` are the raw output logits from the model (shape [batch_size, num_classes, height, width])
        # and `targets` are the ground truth class indices (shape [batch_size, height, width])
        predictions = torch.randn(2, 5, 256, 256, requires_grad=True)  # Example predictions for a batch of 2 images
        targets = torch.randint(0, 5, (2, 256, 256))  # Example targets

        alpha = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)  # Example class weights

        criterion = PixelWiseFocalLoss(alpha=alpha, gamma=2)
        loss = criterion(predictions, targets)
    """
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(PixelWiseFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (batch_size, num_classes, height, width)
        # targets: (batch_size, height, width)
        
        # Compute softmax over the channel dimension
        # softmax = F.softmax(inputs, dim=1)
        
        # Gather the probabilities of the true class
        targets_one_hot = F.one_hot(targets.long(), num_classes=inputs.size(1)).permute(0, 3, 1, 2).float()
        pt = torch.sum(inputs * targets_one_hot, dim=1)  # shape: (batch_size, height, width)
        
        # Compute the focal loss components
        log_pt = torch.log(pt + 1e-10)  # Adding a small value to avoid log(0)
        focal_loss = - (1 - pt) ** self.gamma * log_pt
        
        # Apply class weighting if specified
        if self.alpha is not None:
            if isinstance(self.alpha, (list, torch.Tensor)):
                alpha_t = self.alpha[targets]
            else:
                alpha_t = self.alpha
            focal_loss = alpha_t * focal_loss
        
        # Apply reduction method
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """
    # Example usage:
        combined_criterion = CombinedLoss(alpha=alpha, gamma=2, dice_weight=1.0, focal_weight=1.0)
        loss = combined_criterion(predictions, targets)
        print(loss)
    """
    def __init__(self, alpha=None, gamma=2, reduction='mean', dice_weight=1.0, focal_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.focal_loss = PixelWiseFocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def dice_loss(self, inputs, targets):
        smooth = 1.0
        iflat = inputs.contiguous().view(-1)
        tflat = targets.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

    def forward(self, inputs, targets):
        focal_loss = self.focal_loss(inputs, targets)
        # softmax = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets.long(), num_classes=inputs.size(1)).permute(0, 3, 1, 2).float()
        dice_loss = self.dice_loss(inputs, targets_one_hot)
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss

