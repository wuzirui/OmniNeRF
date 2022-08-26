import torch
from kornia.losses import ssim as dssim

def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred-image_gt)**2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value

def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))

def ssim(image_pred, image_gt, reduction='mean'):
    """
    image_pred and image_gt: (1, 3, H, W)
    """
    dssim_ = dssim(image_pred, image_gt, 3, reduction) # dissimilarity in [0, 1]
    return 1-2*dssim_ # in [-1, 1]

def depth_rmse(
    depth_predict: torch.Tensor,
    depth_gt: torch.Tensor
) -> float:
    """Computes the RMSE between two depths.
    Args:
      depth_predict: torch.tensor. A depth map of size [width, height].
      depth_gt: torch.tensor. A depth map of size [width, height].
    Returns:
      The RMSE between the two depths.
    """
    return torch.sqrt(torch.mean((depth_predict - depth_gt) ** 2))

def depth_rmse_log(
    depth_predict: torch.Tensor,
    depth_gt: torch.Tensor,
    eps: float = 1e-6
) -> float:
    """Computes the RMSE between two depths.
    Args:
      depth_predict: torch.tensor. A depth map of size [width, height].
      depth_gt: torch.tensor. A depth map of size [width, height].
    Returns:
      The RMSE between the two depths.
    """
    return torch.sqrt(torch.mean((torch.log(depth_predict + eps) - torch.log(depth_gt + eps)) ** 2))

def depth_abs_rel(
    depth_predict: torch.Tensor,
    depth_gt: torch.Tensor,
    eps: float = 1e-6
) -> float:
    """Computes the absolute relative error between two depths.
    Args:
      depth_predict: torch.tensor. A depth map of size [width, height].
      depth_gt: torch.tensor. A depth map of size [width, height].
    Returns:
      The absolute relative error between the two depths.
    """
    return torch.mean(torch.abs(depth_predict - depth_gt) / (depth_gt + eps))

def depth_sq_rel(
    depth_predict: torch.Tensor,
    depth_gt: torch.Tensor,
    eps: float = 1e-6
) -> float:
    """Computes the squared relative error between two depths.
    Args:
      depth_predict: torch.tensor. A depth map of size [width, height].
      depth_gt: torch.tensor. A depth map of size [width, height].
    Returns:
      The squared relative error between the two depths.
    """
    return torch.mean((depth_predict - depth_gt) ** 2 / (depth_gt + eps))

def depth_delta(
    depth_predict: torch.Tensor,
    depth_gt: torch.Tensor,
    level: int,
    eps: float = 1e-6
) -> float:
    """Computes the delta error between two depths.
    Args:
      depth_predict: torch.tensor. A depth map of size (width * height).
      depth_gt: torch.tensor. A depth map of size (width * height).
      level: int. The level of the delta error.
    Returns:
      The delta error between the two depths.
    """
    return torch.count_nonzero(torch.max(
        depth_gt / (depth_predict + eps), 
        depth_predict / (depth_gt + eps)) \
             < 1.25 ** level) / depth_gt.numel()