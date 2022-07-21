from numpy import fromstring
import torch
from torch import nn

class ColorLoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)

        return self.coef * loss
               
class DepthLoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['depth_coarse'], targets)
        if 'depth_fine' in inputs:
            loss += self.loss(inputs['depth_fine'], targets)

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
        # before truncation 
        front_mask = torch.where(z_vals < (gt_depth - self.truncation), 
                                torch.ones_like(z_vals),
                                torch.zeros_like(z_vals))
        # after truncation
        back_mask = torch.where(z_vals > (gt_depth + self.truncation),
                                torch.ones_like(z_vals),
                                torch.zeros_like(z_vals))
        # valid depth mask
        depth_mask = torch.where(z_vals > 0,
                                torch.ones_like(z_vals),
                                torch.zeros_like(z_vals))
        # sdf region
        sdf_mask = (1.0 - front_mask) * (1.0 - back_mask) * depth_mask

        # calculate weights for each loss
        freespace_samples = torch.count_nonzero(front_mask)
        sdf_samples = torch.count_nonzero(sdf_mask)
        n_samples = freespace_samples + sdf_samples
        freespace_weight = sdf_samples / n_samples
        sdf_weight = freespace_samples / n_samples
        
        # calculate losses
        freespace_loss = self.img2mse(predicted_sdf * front_mask, 
                                      torch.ones_like(predicted_sdf) * front_mask)
        sdf_loss = self.img2mse((z_vals + predicted_sdf * self.truncation) * sdf_mask,
                                gt_depth * sdf_mask)

        return freespace_weight * freespace_loss, sdf_weight * sdf_loss

class RGBDLoss(nn.Module):
    def __init__(self, color_coef=0.1, depth_coef=0.1, freespace_coef=10, trunc_coef=6000, truncation=0.05):
        super().__init__()
        self.rgb_loss = ColorLoss(color_coef)
        self.depth_loss = DepthLoss(depth_coef)
        self.sdf_loss = SDFLoss(truncation)
        self.color_coef = color_coef
        self.depth_coef = depth_coef
        self.freespace_coef = freespace_coef
        self.trunc_coef = trunc_coef

    def forward(self, input_result, gt_rgb, gt_depth):
        rgb_loss = self.rgb_loss(input_result, gt_rgb)
        depth_loss = self.depth_loss(input_result, gt_depth)
        fs_coarse, sdf_coarse = self.sdf_loss(input_result['z_vals_coarse'],
                                              input_result['weights_coarse'],
                                              gt_depth)
        loss = rgb_loss * self.color_coef + depth_loss * self.depth_coef + \
                fs_coarse * self.freespace_coef + sdf_coarse * self.trunc_coef
        fs_fine, sdf_fine = -1, -1
        if 'z_vals_fine' in input_result:
            fs_fine, sdf_fine = self.sdf_loss(input_result['z_vals_fine'],
                                              input_result['weights_fine'],
                                              gt_depth)
            loss += fs_fine * self.freespace_coef + sdf_fine * self.trunc_coef
        return loss, rgb_loss, depth_loss, fs_coarse, sdf_coarse, fs_fine, sdf_fine
        

loss_dict = {'color': ColorLoss, 'rgbd': RGBDLoss}