from numpy import fromstring
import torch
from torch import nn
from models.sdf_utils import *

class ColorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb_coarse'].view(-1, 3), targets.view(-1, 3))
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'].view(-1, 3), targets.view(-1, 3))

        return loss
               
class DepthLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['depth_coarse'].view(-1, 1), targets.view(-1, 1))
        if 'depth_fine' in inputs:
            loss += self.loss(inputs['depth_fine'].view(-1, 1), targets.view(-1, 1))

        return loss

class SDFLoss(nn.Module):
    def __init__(self, truncation, omni_dir):
        super().__init__()
        self.img2mse = lambda x, y: torch.mean((x - y) ** 2)
        self.truncation = truncation
        self.omni_dir = omni_dir

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
        sdf_samples = torch.count_nonzero(sdf_mask)
        if self.omni_dir:
            fs_mask = front_mask + back_mask
            fs_samples = torch.count_nonzero(fs_mask)
            n_samples = sdf_samples = fs_samples
        else:
            front_samples = torch.count_nonzero(front_mask)
            n_samples = sdf_samples + front_samples
        
        gt_sdf = get_gt_sdf(z_vals, gt_depth, self.truncation, front_mask, back_mask, sdf_mask)

        if self.omni_dir:
            return self.img2mse(predicted_sdf * fs_mask, gt_sdf * fs_mask) / fs_samples, \
                self.img2mse(predicted_sdf * sdf_mask, gt_sdf * sdf_mask) / sdf_samples
        
        return self.img2mse(predicted_sdf * front_mask, gt_sdf * front_mask) * sdf_samples / n_samples, \
            self.img2mse(predicted_sdf * sdf_mask, gt_sdf * sdf_mask) * front_samples / n_samples

class RGBDLoss(nn.Module):
    def __init__(self, color_coef=0.1, depth_coef=0.1, freespace_weight=10, truncation_weight=6000, truncation=0.05, omni_dir=False):
        super().__init__()
        self.rgb_loss = ColorLoss()
        self.depth_loss = DepthLoss()
        self.sdf_loss = SDFLoss(truncation, omni_dir)
        self.color_coef = color_coef
        self.depth_coef = depth_coef
        self.freespace_weight = freespace_weight
        self.truncation_weight = truncation_weight

    def forward(self, input_result, gt_rgb, gt_depth):
        rgb_loss = self.rgb_loss(input_result, gt_rgb)
        depth_loss = self.depth_loss(input_result, gt_depth)
        fs_c, tr_c = self.sdf_loss(input_result['z_vals_coarse'],
                                              input_result['sigmas_coarse'],
                                              gt_depth)
        loss = rgb_loss * self.color_coef + depth_loss * self.depth_coef + fs_c * self.freespace_weight + tr_c * self.truncation_weight
        sdf_fine = -1
        if 'z_vals_fine' in input_result:
            fs_f, tr_f = self.sdf_loss(input_result['z_vals_fine'],
                                              input_result['sigmas_fine'],
                                              gt_depth)
            loss += fs_f * self.freespace_weight + tr_f * self.truncation_weight
        return loss, rgb_loss, depth_loss, fs_c, fs_f, tr_c, tr_f
        

loss_dict = {'color': ColorLoss, 'rgbd': RGBDLoss}