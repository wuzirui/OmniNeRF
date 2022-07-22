import enum
import imp
from logging import root
import torch
import numpy as np
from torch.utils.data import Dataset
import glob
import os
from PIL import Image
from torchvision import transforms as T

from .ray_utils import *

class RGBDDatset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(640, 480), max_val_imgs=20):
        print("Reading RGBD Datset, have a nice day!")
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.white_back = False
        self.max_val_imgs = max_val_imgs
        self.image_transform = T.Compose([
            T.ToTensor(),
        ])
        self.image_paths = glob.glob(os.path.join(root_dir, 'images/*.png'))
        self.depth_paths = glob.glob(os.path.join(root_dir, 'depth/*.png'))
        self.pose_path = os.path.join(root_dir, "poses.txt")
        self.focal_path = os.path.join(root_dir, "focal.txt")
        assert os.path.exists(self.pose_path), "pose file not found"
        self.image_paths = sorted(self.image_paths, key=lambda x: int(x.split('img')[-1].split('.')[0]))
        self.depth_paths = sorted(self.depth_paths, key=lambda x: int(x.split('depth')[-1].split('.')[0]))
        assert len(self.image_paths) == len(self.depth_paths), "Number of images and depths must be equal"
        print("Found %d images with depths" % len(self.image_paths))
        self.read_data()
    
    def read_data(self):
        with open(self.focal_path, 'r') as f:
            self.focal = float(f.readline())
        with open(self.pose_path, 'r') as f:
            self.poses = [np.fromstring(line, sep=' ') for line in f]
            # RGBD dataset has poses already in the form of Twc (camera to world)
            # axes are right up back (OpenGL Coordinate System)
            # we dont need to change the form of the poses
            self.poses = torch.tensor(self.poses, dtype=torch.float32).reshape(-1, 4, 4)
            self.poses = self.poses[:, :3, :]
        if self.split == 'train':
            idx = [i for i in range(len(self.image_paths)) if i % 20 != 0]
            self.n_images = len(idx)
            print(f"selected {len(idx)} images for training")

        elif self.split == 'val':
            idx = [i for i in range(len(self.image_paths)) if i % 20 == 0]
            if self.max_val_imgs is not None and len(idx) > self.max_val_imgs:
                idx = idx[:self.max_val_imgs]
            self.n_images = len(idx)
            print(f"selected {len(idx)} images for validation")
        elif self.split == 'test_train':
            idx = [i for i in range(len(self.image_paths)) if i % 20 != 0][:10]
            self.n_images = len(idx)
            print(f"selected {len(idx)} images for training")
        else:
            raise ValueError("Split must be either train, test_train or val")
        self.image_paths = [self.image_paths[i] for i in idx]
        self.depth_paths = [self.depth_paths[i] for i in idx]
        self.poses = self.poses[idx]
        self.all_rgbs = []
        self.all_depths = []
        self.bounds = []
        for i, (image_path, depth_path) in enumerate(zip(self.image_paths, self.depth_paths)):
            rgb = Image.open(image_path).convert('RGB')
            depth = Image.open(depth_path)
            # rgb = self.image_transform(rgb) \
            rgb = self.image_transform(rgb.resize(self.img_wh, Image.LANCZOS)) \
                      .view(3, -1) \
                      .permute(1, 0)
            # depth = self.image_transform(depth) \
            depth = self.image_transform(depth.resize(self.img_wh, Image.LANCZOS))\
                        .view(1, -1) \
                        .permute(1, 0)
            # rgb (h * w, 3) depth (h * w, 1)
            bound_min, bound_max = depth.min(), depth.max()
            self.all_rgbs += [rgb]
            self.all_depths += [depth]
            self.bounds += [[bound_min, bound_max]]

        if self.split == 'train' or self.split == 'test_train':
            self.all_rgbs = torch.cat(self.all_rgbs, dim=0)  # (N * H * W, 3)
            self.all_depths = torch.cat(self.all_depths, dim=0).float()
            # (N * H * W, 1)
            assert self.all_rgbs.shape == (self.n_images * self.img_wh[0] * self.img_wh[1], 3), \
                f"RGB shape is {self.all_rgbs.shape}, expected {(self.n_images * self.img_wh[0] * self.img_wh[1], 3)}"
        else:
            self.all_rgbs = torch.stack(self.all_rgbs, dim=0)  # (N, H * W, 3)
            self.all_depths = torch.stack(self.all_depths, dim=0).float()
            # (N, H * W, 1)

        self.bounds = torch.tensor(self.bounds, dtype=torch.float32) / 1000.0
        self.all_depths = np.array(self.all_depths) / 1000.0
        # scale depth to meters
        scale_factor = 0.25
        self.all_depths *= scale_factor
        self.poses[..., 3] *= scale_factor

        self.bounds *= scale_factor

        print(f"done loading images")
        print(f"precomputing rays")

        # same for all images. Generate a set of rays in camera coordinate system
        # according to its size and focal length (intrinsic parameters)
        # ray origin at the center of the image
        self.directions = get_ray_directions(self.img_wh[1], self.img_wh[0], self.focal)
        # (H, W, 3)

        self.all_rays = []
        for c2w in self.poses:
            # convert precomputed camera rays (in camera coordinate system) 
            # to world coordinate system
            rays_o, rays_d = get_rays(self.directions, c2w)
            # both (H * W, 3)
            
            # RGBD datasets are taken in spheric inward-facing manner by default
            # so we dont use NDC space, which is used in forward-facing datasets

            # ray format: (H * W, 8), foreach ray: 
            # origin(3), direction(3), near bound(1), far bound(1)
            near, far = 0, 1
            self.all_rays += [torch.cat([
                rays_o, rays_d,
                near * torch.ones_like(rays_o[:, :1]),
                far * torch.ones_like(rays_o[:, :1])
            ], -1)]
        if self.split == 'train' or self.split == 'test_train':
            self.all_rays = torch.cat(self.all_rays, dim=0)  # (N * H * W, 8)
        else:
            self.all_rays = torch.stack(self.all_rays, dim=0)
            # (N, H * W, 8)
        print(f"done")

        

    def __len__(self):
        return len(self.all_rays)

    def __getitem__(self, idx):
        return {
            'rays': self.all_rays[idx],
            'rgbs': self.all_rgbs[idx],
            'depths': self.all_depths[idx]
        }