from src.datamanager import *
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from . import features
from . import metric


class HardTripletSampler(DataProvider):
    def __init__(self, dataset, loader, net, metric, num_neg=10, num_pos=3,
                 num_random_samples=5000, hard_sampling=True,
                 layer_embeddings=('emb\\model\\features\\features_pool'),
                 sample_size=None, keep_aspect_ratio=True, transform=None, transform_target=None,
                 return_only_image=True):
        super(HardTripletSampler, self).__init__(dataset, loader, sample_size=None, keep_aspect_ratio=keep_aspect_ratio,
                                                 transform=transform, transform_target=transform_target, return_only_image=return_only_image)

        self.net = net
        self.metric = metric
        self.layer_embeddings = layer_embeddings
        self.D = np.array([])
        self.probe_info = []
        self.gallery_info = []
        self.num_neg = num_neg
        self.num_pos = num_pos
        self.num_random_samples = num_random_samples
        self.hard_sampling = hard_sampling
        self.random_subset = None
        self.anchor_indexes = []
        self.transform_sample_size = sample_size

    def __len__(self):
        return self.num_random_samples

    def _compute_distances(self):

        # Random query indexes
        self.anchor_indexes = np.sort(np.random.choice(len(self.dataset), self.num_random_samples, replace=False))

        # Get subset
        self.random_subset = self.dataset.extract_subset(self.anchor_indexes)

        # Hard sampling?
        if self.hard_sampling:

            # Do not apply any data augmentation to compute the feature distance for hard mining!
            transformations = DataTransformer([
                transforms.Resize(self.transform_sample_size, interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                self.transform.transformations[-1]])

            # Init new data provider
            dp = DataProvider(self.random_subset, self.loader, sample_size=self.sample_size, keep_aspect_ratio=self.keep_aspect_ratio,
                              transform=transformations, transform_target=self.transform_target, return_only_image=self.return_only_image)

            # Get features
            X = features.get_features(self.net, [dp], self.layer_embeddings, batch_size=256, workers=2)

            # Get all pairwise dissimilarities
            self.D, _, self.probe_info, self.gallery_info = metric.get_distance(self.random_subset, X, self.metric, probe=range(0,len(self.anchor_indexes)), gallery=range(0,len(self.anchor_indexes)), re_rank=False)

    def _get_pos_neg(self, anchor_index, anchor_target):
        if self.hard_sampling:
            g_id = np.array(self.gallery_info[1])
            d = self.D[anchor_index, :]

            # Pos
            # Sample that has the maximum distance (biggest error)
            pos_idx = np.where(g_id == anchor_target)[0]
            #pos_idx = pos_idx[np.argmax(d[pos_idx])]
            pos_idx = pos_idx[np.argsort(d[pos_idx])[::-1]]
            idx = np.random.choice(min(len(pos_idx), self.num_pos), 1)[0]
            pos_im, pos_target, *args = self.random_subset[pos_idx[idx]]

            # Neg
            # Pick the N samples that have the minimum distance (smallest error)
            n = self.num_neg
            neg_idx = np.where(g_id != anchor_target)[0]
            neg_idx = neg_idx[np.argsort(d[neg_idx])][:n]
            idx = np.random.choice(n, 1)[0]
            neg_im, neg_target, *args = self.random_subset[neg_idx[idx]]
        else:
            pos_idx = np.random.choice(np.where(np.array(self.random_subset.targets) == anchor_target)[0])
            pos_im, pos_target, *args = self.random_subset[pos_idx]
            neg_idx = np.random.choice(np.where(np.array(self.random_subset.targets) != anchor_target)[0])
            neg_im, neg_target, *args = self.random_subset[neg_idx]

        return pos_im, pos_target, neg_im, neg_target

    def update(self, *args, **kwargs):
        # No need to udpate during evaluation
        if len(kwargs) > 0 and 'is_train' in kwargs and not kwargs['is_train']:
            return
        self._compute_distances()

    def get_sample(self, anchor_index):

        # anchor_index is in [0, self.num_random_samples-1]
        im, target, *args = self.random_subset[anchor_index]
        anchor = self.loader(im, self.sample_size, self.keep_aspect_ratio)

        # Get pos and neg
        pos_im, pos_target, neg_im, neg_target = self._get_pos_neg(anchor_index, target)
        pos = self.loader(pos_im, self.sample_size, self.keep_aspect_ratio)
        neg = self.loader(neg_im, self.sample_size, self.keep_aspect_ratio)

        if self.transform is not None:
            anchor = self.transform(anchor)
            pos = self.transform(pos)
            neg = self.transform(neg)

        if self.transform_target is not None:
            target = self.transform_target(target)

        if self.return_only_image:
            return (anchor, pos, neg), (target, pos_target, neg_target)
        return (anchor, pos, neg), (target, pos_target, neg_target), im


