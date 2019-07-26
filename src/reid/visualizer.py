import numpy as np
from PIL import Image
from ..configs import DatasetConfig
from ..datamanager import *
from ..visualization import visualizer


def display_dataset_strip(dataset_name, cams, person_ids, file_name, imsize=(256,128), samples_from_cam=1):
    dset_opts = DatasetConfig(dataset_name, imsize, (0.5, 0.5))
    dset, _, data_loader = get_dataset(dset_opts, save=False)

    data_provider = DataProvider(dset, sample_size=imsize, loader=data_loader, return_only_image=False, keep_aspect_ratio=False)

    indexes = []
    for person_id in person_ids:
        person_indexes = []
        for cam in cams:
            idx, _, _ = dset.get_indexes_from_id_cam(person_id, cam)
            if len(idx) > 0:
                person_indexes.extend(idx[:samples_from_cam])
        indexes.extend(person_indexes)

    return visualizer.display_dataset_images_in_grid(data_provider, image_indexes=indexes, file_name=file_name,
                                              nrows=int(len(indexes)/len(person_ids)))