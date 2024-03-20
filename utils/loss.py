import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch 

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()
        
        
class BCEWithLogitsLossWithIgnoreIndex(nn.Module):
    def __init__(self, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets, weight=None):
        # inputs of size B x C x H x W
        n_cl = torch.tensor(inputs.shape[1]).to(inputs.device)
        labels_new = torch.where(targets != self.ignore_index, targets, n_cl)
        # replace ignore with numclasses + 1 (to enable one hot and then remove it)
        targets = F.one_hot(labels_new, inputs.shape[1] + 1).float().permute(0, 3, 1, 2)
        targets = targets[:, :inputs.shape[1], :, :]  # remove 255 from 1hot
        # targets is B x C x H x W so shape[1] is C

        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # loss has shape B x C x H x W
        # sum the contributions of the classes
        
        if weight is not None:
            weight = weight.reshape(1, -1, 1, 1)[:, :inputs.shape[1], :, :]
            loss = loss * weight
            loss = loss/weight.sum()
            loss = loss.sum(dim=1)
        else:
            loss = loss.sum(dim=1)
        
        if self.reduction == 'mean':
            #return loss.mean()
            # if targets have only zeros, we skip them
            return torch.masked_select(loss, targets.sum(dim=1) != 0).mean()
        elif self.reduction == 'sum':
            #return loss.sum()
            return torch.masked_select(loss, targets.sum(dim=1) != 0).sum()
        else:
            #return loss
            return loss * targets.sum(dim=1)


class WBCELoss(nn.Module):
    def __init__(self, ignore_index=255, pos_weight=None, reduction='mean', n_old_classes=0, n_new_classes=0):
        super().__init__()
        self.ignore_index = ignore_index
        self.n_old_classes = n_old_classes  # |C0:t-1| + 1(bg), 19-1: 20 | 15-5: 16 | 15-1: 16...
        self.n_new_classes = n_new_classes  # |Ct|, 19-1: 1 | 15-5: 5 | 15-1: 1

        pos_weight = pos_weight[n_old_classes: n_old_classes + n_new_classes]
        self.reduction = reduction
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=self.reduction)

    def forward(self, logit, label):
        # logit:     [N, |Ct|, H, W]
        # label:     [N, H, W]

        N, C, H, W = logit.shape
        target = torch.zeros_like(logit, device=logit.device).float()
        for cls_idx in label.unique():
            if cls_idx in [0, self.ignore_index]:
                continue
            target[:, int(cls_idx) - self.n_old_classes] = (label == int(cls_idx)).float()

        loss = self.criterion(
            logit.permute(0, 2, 3, 1).reshape(-1, C),
            target.permute(0, 2, 3, 1).reshape(-1, C)
        )

        if self.reduction == 'none':
            return loss.reshape(N, H, W, C).permute(0, 3, 1, 2)  # [N, C, H, W]
        elif self.reduction == 'mean':
            return loss
        else:
            raise NotImplementedError


class KDLoss(nn.Module):
    def __init__(self, pos_weight=None, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logit, logit_old=None):
        # logit:     [N, |Ct|, H, W]
        # logit_old: [N, |Ct|, H, W]

        # N, C, H, W = logit.shape
        # loss = self.criterion(
        #     logit.permute(0, 2, 3, 1).reshape(-1, C),
        #     logit_old.permute(0, 2, 3, 1).reshape(-1, C)
        # )
        logit = torch.log_softmax(logit, dim=1)
        loss = -(logit * logit_old).sum(dim=1)
        # if self.reduction == 'none':
        #     return loss.reshape(N, H, W, C).permute(0, 3, 1, 2)  # [N, C, H, W]
        #
        # elif self.reduction == 'mean':
        #     return loss
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss


class ACLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.criterion = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, logit, label):
        # logit: [N, 1, H, W]
        target = (label == 0).float().unsqueeze(dim=1)
        return self.criterion(logit, target)

class Dynamic_Loss(nn.Module):
    def __init__(self, num_classes, alpha=0.99):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=255)
        self.past_classes = sum(num_classes[:-1])
        self.num_classes = sum(num_classes)
        self.alpha = alpha
        self.moving_prob_avg = np.array([-1 for i in range(self.num_classes)], dtype=float)

    def forward(self, logit, ema_prob, ema_thresh, real_labels):
        # calculate current probilites and label
        logit_prob = logit.detach().softmax(dim=1)
        # scores, labels = logit_prob.max(dim=1)
        _, labels = logit_prob[:, :self.past_classes].max(dim=1)
        labels = torch.where(real_labels >= self.past_classes, real_labels, labels)
        index = torch.where((real_labels == 255), 0, labels)
        scores = torch.gather(logit_prob, dim=1, index=index.unsqueeze(1)).squeeze(1)
        for label in torch.unique(labels):
            if label == 255:
                continue
            if self.moving_prob_avg[label] == -1:
                self.moving_prob_avg[label] = scores[labels == label].mean().item()
            else:
                self.moving_prob_avg[label] = (1 - self.alpha) * scores[labels == label].mean().item() + \
                                              self.alpha * self.moving_prob_avg[label]
        dynamic_thresh = []
        for i in range(len(ema_thresh)):
            if self.moving_prob_avg[i] != -1:
                dynamic_thresh.append(ema_thresh[i] * self.moving_prob_avg[i])
            else:
                dynamic_thresh.append(ema_thresh[i])

        ema_mask = torch.tensor(0).cuda()
        # ema_scores, ema_labels = torch.max(ema_prob, dim=1)
        _, ema_labels = torch.max(ema_prob[:, :self.past_classes], dim=1)
        ema_labels = torch.where(real_labels >= self.past_classes, real_labels, ema_labels)
        index = torch.where((real_labels == 255), 0, ema_labels)
        ema_scores = torch.gather(ema_prob, dim=1, index=index.unsqueeze(1)).squeeze(1)
        for i in range(self.num_classes):
            ema_mask = ema_mask | ((ema_labels == i) & (ema_scores >= ema_thresh[i]))
        dynamic_labels = torch.where(ema_mask>0, ema_labels, 255)

        losses = self.criterion(logit, dynamic_labels)
        for dynamic_label in torch.unique(dynamic_labels):
            if dynamic_label == 255:
                continue
            losses[dynamic_labels == dynamic_label] *= 1 - self.moving_prob_avg[dynamic_label]

        return losses[dynamic_labels != 255].mean()



# class KDLoss2(nn.Module):
#     def __init__(self, pos_weight=None, reduction='mean'):
#         super().__init__()
#         self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)
#
#     def forward(self, logit, logit_old):
#         # logit:     [N, |Ct|, H, W]
#         # logit_old: [N, |Ct|, H, W]
#         loss = self.criterion(logit, logit_old)
#         return loss