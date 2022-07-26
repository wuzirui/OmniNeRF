from .blender import BlenderDataset
from .llff import LLFFDataset
from .rgbd_utils import RGBDDatset

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'rgbd': RGBDDatset}