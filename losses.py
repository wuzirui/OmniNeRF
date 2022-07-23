from numpy import fromstring
import torch
from torch import nn
from models.sdf_utils import *

class ColorLoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb_coarse'].view(-1, 3), targets.view(-1, 3))
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'].view(-1, 3), targets.view(-1, 3))

        return self.coef * loss
               
class DepthLoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['depth_coarse'].view(-1, 1), targets.view(-1, 1))
        if 'depth_fine' in inputs:
            loss += self.loss(inputs['depth_fine'].view(-1, 1), targets.view(-1, 1))

        return self.coef * loss

class SDFLoss(nn.Module):
    def __init__(self, truncation):
        super().__init__()
        self.img2mse = lambda x, y: torch.mean((x - y) ** 2)
        self.truncation = truncation

    def forward(self, z_vals, predicted_sdf, gt_depth):
        """
        calculate SDF losses, consists of two parts:
        1. freespace sdf loss, includes before/after truncation region
        2. truncation loss
        in this function, we first compute masks for the truncation region
        and compute losses respectively

        Inputs:
            z_vals: (batch_size, n_samples)
            predicted_sdf: (batch_size, n_samples)
            gt_depth: (batch_size, 1)
        """
        gt_depth = gt_depth[:, None]
        front_mask, back_mask, sdf_mask = get_gt_sdf_masks(z_vals, gt_depth, self.truncation)
        
        gt_sdf = get_gt_sdf(z_vals, gt_depth, self.truncation, front_mask, back_mask, sdf_mask)

        return self.img2mse(predicted_sdf, gt_sdf)

class RGBDLoss(nn.Module):
    def __init__(self, color_coef=0.1, depth_coef=0.1, sdf_coef=10, truncation=0.05):
        super().__init__()
        self.rgb_loss = ColorLoss(color_coef)
        self.depth_loss = DepthLoss(depth_coef)
        self.sdf_loss = SDFLoss(truncation)
        self.color_coef = color_coef
        self.depth_coef = depth_coef
        self.sdf_coef = sdf_coef

    def forward(self, input_result, gt_rgb, gt_depth):
        rgb_loss = self.rgb_loss(input_result, gt_rgb)
        depth_loss = self.depth_loss(input_result, gt_depth)
        sdf_coarse = self.sdf_loss(input_result['z_vals_coarse'],
                                              input_result['sigmas_coarse'],
                                              gt_depth)
        loss = rgb_loss + depth_loss + sdf_coarse * self.sdf_coef
        sdf_fine = -1
        if 'z_vals_fine' in input_result:
            sdf_fine = self.sdf_loss(input_result['z_vals_fine'],
                                              input_result['sigmas_fine'],
                                              gt_depth)
            loss += sdf_fine * self.sdf_coef
        return loss, rgb_loss, depth_loss, sdf_coarse, sdf_fine
        

loss_dict = {'color': ColorLoss, 'rgbd': RGBDLoss}