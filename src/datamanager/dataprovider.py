import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from multiprocessing import Pool
import time
from joblib import Parallel, delayed


class DataProvider(data.Dataset):
    def __init__(self, dataset, loader, sample_size=None, keep_aspect_ratio=True, transform=None, transform_target=None, return_only_image=True):
        self.sample_size = sample_size
        self.keep_aspect_ratio = keep_aspect_ratio
        self.transform = transform
        self.transform_target = transform_target
        self.dataset = dataset
        self.loader = loader
        self.return_only_image = return_only_image

    def get_sample(self, index):
        # *args has been inlcuded to ignore other information when using particular datasets, such as the ReID ones..
        im, target, *args = self.dataset[index]
        img = self.loader(im, self.sample_size, self.keep_aspect_ratio)

        if self.transform is not None:
            img = self.transform(img)
        if self.transform_target is not None:
            target = self.transform_target(target)

        if self.return_only_image:
            return img, target
        return img, target, im

    def get_samples(self, indexes, n_jobs=1):

        # Load all images
        if n_jobs > 1:
            images = Parallel(n_jobs=n_jobs)( delayed(self.get_sample)(ii) for ii in indexes)
        else:
            images = [self.get_sample(ii) for ii in indexes]
        return images

    def __getitem__(self, index):
        return self.get_sample(index)

    def __len__(self):
        return len(self.dataset.images)

    def get_mean_and_std(self, n_samples=None, n_jobs=20):
        """
        Calculates the mean and the std of the dataset

        Returns:
            list of floats, the mean of the dataset on the different channels
        """
        pt = time.time()

        # How many samples to get mean and std?
        if n_samples is None:
            n_samples = self.dataset.length

        # Range over all samples to consider
        if n_jobs > 1:
            mean_std = Parallel(n_jobs=n_jobs)(delayed(self.get_sample_mean_std)(ii) for ii in range(n_samples))
        else:
            mean_std = [self.get_sample_mean_std(idx) for idx in range(n_samples)]

        # Average over all samples
        mean_std = np.mean(np.array(mean_std), axis=0)

        # Get average mean and std
        mean = mean_std[0].astype(float)
        std = mean_std[1].astype(float)

        print('==> Mean: {} - Std: {} -- {:.2f}'.format(mean.copy(), std.copy(), time.time()-pt))
        return mean, std

    def get_sample_mean_std(self, sample_idx):
        """
        Compute the mean and standard deviation (per channel) of the image identified by the given sample index 
        :param sample_idx: sample index from which extract mean and standard deviation per channel 
        :return: mean
                 std
        """
        sample = self.get_sample(sample_idx)[0]
        if isinstance(sample, np.ndarray):
            img = sample
        elif isinstance(sample, torch.Tensor):
            img = sample.numpy().transpose(1,2,0)
        else:
            img = np.array(sample)
        mean = np.mean(img, axis=(0,1))
        std = np.std(img, axis=(0, 1))
        return mean, std



