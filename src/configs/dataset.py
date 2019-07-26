from .base import BaseConfig

class DatasetConfig(BaseConfig):
    def __init__(self, name, imsize, split, cam_pair=None, samples_per_class=9999999):
        super(DatasetConfig, self).__init__()
        self.split = split
        self.name = name
        self.imsize = imsize
        self.camera_pair = cam_pair
        self.samples_per_class = samples_per_class

