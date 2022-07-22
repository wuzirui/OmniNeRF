from math import trunc
import torch

def get_gt_sdf_masks(z_vals, gt_depth, truncation):
    """
    Inputs:
        z_vals: (batch_size, n_samples)
        gt_depth: (batch_size, 1)
        truncation: float
    """
    # before truncation 
    front_mask = torch.where(z_vals < (gt_depth - truncation), 
                            torch.ones_like(z_vals),
                            torch.zeros_like(z_vals))
    # after truncation
    back_mask = torch.where(z_vals > (gt_depth + truncation),
                            torch.ones_like(z_vals),
                            torch.zeros_like(z_vals))
    # sdf region
    sdf_mask = (1.0 - front_mask) * (1.0 - back_mask)

    # calculate weights for each loss
    freespace_samples = torch.count_nonzero(front_mask)
    sdf_samples = torch.count_nonzero(sdf_mask)
    n_samples = freespace_samples + sdf_samples
    freespace_weight = sdf_samples / n_samples
    sdf_weight = freespace_samples / n_samples
    return front_mask, back_mask, sdf_mask, freespace_weight, sdf_weight

def get_gt_sdf(z_vals, gt_depth, truncation, front_mask, back_mask, sdf_mask):
    """
    Inputs:
        z_vals: (batch_size, n_samples)
        gt_depth: (batch_size, 1)
        truncation: float
    """
    fs_sdf = -front_mask + back_mask                          # freespace
    tr_sdf = sdf_mask * (z_vals - gt_depth) / truncation      # sdf
    return fs_sdf + tr_sdf

