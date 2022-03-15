from torch import device, nn
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys


class Loss(nn.Module):
    def __init__(self, S, B, lambda_coord, lambda_noobj, device) -> None:
        super().__init__()
        self.S = S
        self.B = B
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.device = device

    def calculate_location_loss(self, predictions, targets):
        xy_mask = torch.zeros(predictions.shape, dtype=torch.bool).to(self.device)
        for i in range(self.B):
            xy_mask[:, i*5:i*5+2] = True
        xy_predictions = predictions[xy_mask].view(-1, 2*self.B)
        xy_targets = targets[xy_mask].view(-1, 2*self.B)
        xy_loss = F.mse_loss(xy_predictions, xy_targets, reduction='sum')

        wh_mask = torch.zeros(predictions.shape, dtype=torch.bool).to(self.device)
        for i in range(self.B):
            wh_mask[:, i*5+2:i*5+4] = True
        wh_predictions = predictions[wh_mask].view(-1, 2*self.B)
        wh_targets = predictions[wh_mask].view(-1, 2*self.B)
        wh_predictions = torch.abs(wh_predictions)
        wh_targets = torch.abs(wh_targets)
        wh_loss = F.mse_loss(torch.sqrt(wh_predictions), torch.sqrt(wh_targets), reduction='sum')

        return xy_loss+wh_loss

    def calculate_iou(self, a, b):
        ax = a[0]/self.S
        ay = a[1]/self.S
        a_width_half = a[2]/2
        a_height_half = a[3]/2

        ax_min = ax - a_width_half
        ay_min = ay - a_height_half
        ax_max = ax + a_width_half
        ay_max = ay + a_height_half

        bx = b[0]/self.S
        by = b[1]/self.S
        b_width_half = b[2]/2
        b_height_half = b[3]/2

        bx_min = bx - b_width_half
        by_min = by - b_height_half
        bx_max = bx + b_width_half
        by_max = by + b_height_half

        intersection_xmin = torch.max(ax_min, bx_min)
        intersection_ymin = torch.max(ay_min, by_min)
        intersection_xmax = torch.min(ax_max, bx_max)
        intersection_ymax = torch.min(ay_max, by_max)
        intersection_width = intersection_xmax-intersection_xmin
        intersection_height = intersection_ymax-intersection_ymin
        if intersection_width <= 0 or intersection_height <= 0:
            return 0.0

        intersection_area = intersection_width*intersection_height
        a_area = (ax_max-ax_min)*(ay_max-ay_min)
        b_area = (bx_max-bx_min)*(by_max-by_min)

        iou = intersection_area/(a_area+b_area-intersection_area)
        return iou

    def calculate_confidence_loss(self, predictions, targets):
        confidence_predictions = torch.zeros((predictions.shape[0], self.B)).to(self.device)
        confidence_targets = torch.zeros((targets.shape[0], self.B)).to(self.device)
        for i in range(self.B):
            confidence_predictions[:, i] = predictions[:, i*5+4]
            confidence_targets[:, i] = targets[:, i*5+4]

        return F.mse_loss(confidence_predictions, confidence_targets, reduction='sum')

    def choose_responsible_box(self, predictions, targets):
        for i, prediction, target in zip(range(targets.shape[0]), predictions, targets):
            ious = torch.zeros(self.B)
            for j in range(self.B):
                iou = self.calculate_iou(prediction[j*5:j*5+4], target[j*5:j*5+4])
                ious[j] = iou
            max_iou_index = torch.argmax(ious)
            for j in range(self.B):
                if j == max_iou_index:
                    continue

                targets[i, j*5:j*5+5] = 0

        return targets

    def calculate_classify_loss(self, predictions, targets):
        class_predictions = predictions[:, self.B*5:]
        class_targets = targets[:, self.B*5:]
        return F.mse_loss(class_predictions, class_targets, reduction='sum')

    def forward(self, predictions, targets):
        if predictions.shape != targets.shape:
            os.exit('predictions shape not equal targets shape')

        obj_mask = (targets[:, :, :, 4] > 0).unsqueeze(-1).expand_as(targets)
        obj_predictions = predictions[obj_mask].view(-1, 30)
        obj_targets = targets[obj_mask].view(-1, 30)
        obj_targets = self.choose_responsible_box(obj_predictions, obj_targets)

        # location loss
        loss = torch.tensor(0.0, dtype=torch.float).to(self.device)
        loss += self.lambda_coord*self.calculate_location_loss(obj_predictions, obj_targets)

        # confidence loss
        obj_confidence_loss = self.calculate_confidence_loss(obj_predictions, obj_targets)
        loss += obj_confidence_loss

        noobj_mask = (targets[:, :, :, 4] == 0).unsqueeze(-1).expand_as(targets)
        noobj_predictions = predictions[noobj_mask].view(-1, 30)
        noobj_targets = targets[noobj_mask].view(-1, 30)
        noobj_confidence_loss = self.calculate_confidence_loss(noobj_predictions, noobj_targets)
        loss += self.lambda_noobj * noobj_confidence_loss

        # classify loss
        loss += self.calculate_classify_loss(obj_predictions, obj_targets)

        loss /= predictions.shape[0]
        return loss


if __name__ == '__main__':
    predictions = torch.zeros((1, 7, 7, 30))
    data = np.load('./data/targets/2007_000027.jpg.npy')
    targets = torch.from_numpy(data).unsqueeze(0)

    loss_function = Loss(7, 2, 5, 0.5, 'cpu')
    loss = loss_function(predictions, targets)
    print(f'get yolo loss: {loss.item()}')
    os.system('pause')
