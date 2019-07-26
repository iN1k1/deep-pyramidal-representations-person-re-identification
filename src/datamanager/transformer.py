import numpy as np
from PIL import Image
import random
import math


class DataTransformer(object):
    """
    Defines an object that is responsible to transform the
    example images

    img: numpy array
    """

    def __init__(self, transformations):
        super(DataTransformer, self).__init__()
        self.transformations = transformations

        self._clean()

    def __call__(self, img):
        return self.transform(img)

    def transform(self, img, transformations=None):
        if transformations is None:
            transformations = self.transformations
        return self.apply_transformations(img, transformations)

    def add_transformation(self, transformation):
        self.transformations.append(transformation)

    def apply_transformations(self, img, transformations):
        """
        Args:
            img: a image matrix

        Returns:
            a image matrix with the transformation applied
        """
        for trsf in transformations:
            img = trsf(img)

        return img

    def _clean(self):
        self.transformations = [trsf for trsf in self.transformations if trsf is not None]


class ToNumpy(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return np.array(img)


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.2, r1=0.3, r2=3.3333, max_trials=100, random_values=None):
        self.probability = probability
        #self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r2 = r2
        self.max_trials = max_trials
        self.random_values = random_values

    def __call__(self, img):

        if np.random.choice(2, 1, p=[1- self.probability,  self.probability])[0] == 0:
            return img

        img_w, img_h = img.size
        img_area = img_w * img_h

        img_channels = 3 if img.mode == 'RGB' else 1

        # Generate random values in [0,255] for each channel
        random_pixel_values = self.random_values
        if random_pixel_values is None:
            random_pixel_values = tuple(random.randint(0, 255) for _ in range(img_channels))

        # Attempts
        for attempt in range(self.max_trials):

            # Target are and aspect ratio
            target_area = random.uniform(self.sl, self.sh) * img_area
            aspect_ratio = random.uniform(self.r1, self.r2)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            x = random.randint(0, w)
            y = random.randint(0, h)

            if x+w <= img_w and y+h <= img_h:
                paste_img = Image.new(img.mode, (w,h), color=random_pixel_values)
                img.paste(paste_img, box=(x,y))
                return img

        return img

