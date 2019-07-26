from ..datamanager import utils as datautils
import pickle
import os
import numpy as np
import sklearn.preprocessing as preprocess
from multiprocessing import Pool
from functools import partial


class FeatureExtractor(object):

    def __init__(self, dense=None, normalization=()):
        self.dense = dense
        self.type = ''
        self.normalizations = normalization
        self.parallel = False

    def extract(self, numpy_image, apply_normalizations=False):
        pass

    def extract_all(self, dset, main_path, apply_normalizations=False):
        path = self.get_file_path(main_path)
        if os.path.exists(path):
            features = self.load(main_path)
        else:
            features = [None] * len(dset)
            if self.parallel:
                with Pool(processes=20) as pool:
                    features = pool.map(partial(self.extract_idx, dset=dset, apply_normalizations=apply_normalizations), range(len(dset)))
            else:
                for ii in range(0, len(dset)):
                    im = datautils.pil2np(dset[ii][0])
                    features[ii] = self.extract(im, apply_normalizations=apply_normalizations)
            self.save(features, main_path)
        return features

    def extract_idx(self, idx, dset, apply_normalizations=False):
        im = datautils.pil2np(dset[idx][0])
        return self.extract(im, apply_normalizations=apply_normalizations)

    def feat2np(self, feat):
        return feat[0]

    def get_file_path(self, main_path):
        return main_path

    def save(self, X, main_path):
        path = self.get_file_path(main_path)
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(path, 'wb') as fp:
            pickle.dump(X, fp)

    def load(self, main_path):
        path = self.get_file_path(main_path)
        with open(path, 'rb') as f:
            X = pickle.load(f)
        return X

    def normalize(self, numpy_feat, norm_types=None):
        if norm_types is None:
            norm_types = self.normalizations
        for norm_type in norm_types:
            if norm_type == 'l2' or norm_type == 'l1' or norm_type == 'max':
                numpy_feat = preprocess.normalize(numpy_feat, norm_type, axis=1)
            elif norm_type == 'power':
                numpy_feat = np.sign(numpy_feat) * np.sqrt(np.abs(numpy_feat))
        return numpy_feat



