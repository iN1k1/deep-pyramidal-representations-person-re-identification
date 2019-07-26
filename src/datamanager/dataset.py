import pickle
import os
import numpy as np
from .utils import make_dataset_images, find_classes
from operator import itemgetter
import copy


class Dataset(object):
    def __init__(self, name, root_folder, im_size=None, in_memory=False):
        super(Dataset, self).__init__()
        self.name = name
        self.images = []
        self.targets = []
        self.root = root_folder
        self.length = 0
        self.im_size = im_size
        self.classes = []
        self.class_to_idx = []
        self.idx_to_class = []
        self.in_memory = in_memory
        self.data_path = os.path.join(root_folder, self.name + '.dat')
        self._compute_data_path()

    def _compute_data_path(self):
        if self.im_size is not None:
            self.data_path = os.path.join(self.root,'{}_sz{}_mem{}.dat'.format(self.name, "%s_%s" % self.im_size, self.in_memory))
        else:
            self.data_path = os.path.join(self.root,'{}_mem{}.dat'.format(self.name, self.in_memory))

    def load(self, path=None):
        if path is None:
            path = self.data_path
        if os.path.exists(path):
            data = []
            with open(path, 'rb') as f:
                for _ in range(pickle.load(f)):
                    data.append(pickle.load(f))
            self.images, self.targets, self.classes, self.class_to_idx = data
        else:
            self.classes, self.class_to_idx = find_classes(self.root)
            dset = make_dataset_images(self.root, self.class_to_idx)
            self.images = [dset[ii][0] for ii in range(0, len(dset))]
            self.targets = [dset[ii][1] for ii in range(0, len(dset))]

        self.compute_idx_to_class()
        self.length = len(self.targets)

    def save(self, path=None):
        if path is None:
            path = self.data_path
        data = [self.images, self.targets, self.classes, self.class_to_idx]
        with open(path, 'wb') as fp:
            pickle.dump(len(data), fp)
            for value in data:
                pickle.dump(value, fp)

    def clone(self, clear_data=False):
        clone = copy.deepcopy(self)
        if clear_data:
            clone.images = []
            clone.targes = []
            clone.length = 0
        return clone

    def compute_idx_to_class(self):
        self.idx_to_class = {v: k for v, k in zip(list(self.class_to_idx.values()), list(self.class_to_idx.keys()))}
        return self.idx_to_class

    def extract_subset(self, idx, dset=None):
        if dset is None:
            dset = Dataset(self.name, self.root, self.im_size)
        dset.classes = copy.copy(self.classes)
        dset.class_to_idx = copy.copy(self.class_to_idx)
        dset.idx_to_class = copy.copy(self.idx_to_class)
        if len(idx) > 0:
            dset.images = itemgetter(*idx)(self.images)
            dset.targets = itemgetter(*idx)(self.targets)
            if isinstance(dset.targets, int):
                dset.targets = [dset.targets]
                dset.images = [dset.images]
            dset.length = len(dset.targets)

        return dset

    def append_subset(self, dset, indexes=None, create_new_dset=False):
        if indexes is None:
            indexes = range(len(dset))

        # Get new dataset that need to be added
        dset_to_add = dset.extract_subset(indexes)

        # Orig dset
        dset_orig = self
        if create_new_dset:
            dset_orig = self.clone(clear_data=True)

        # Extend data containers
        dset_orig.images.extend(dset_to_add.images)
        dset_orig.targets.extend(dset_to_add.targets)
        dset_orig.length = len(self.targets)

        return dset_orig

    def diff_subset(self, dset, indexes=None, create_new_dset=False):
        # TODO
        pass
        # if indexes is None:
        #     indexes = range(len(dset))
        #
        # # Get new dataset with data that need to be removed
        # dset_to_removed = dset.extract_subset(indexes)
        #
        # # Orig dset
        # dset_orig = self
        # if create_new_dset:
        #     dset_orig = self.clone(clear_data=True)
        #
        # # Extend data containers
        # dset_orig.images.extend(dset_to_add.images)
        # dset_orig.targets.extend(dset_to_add.targets)
        # dset_orig.length = len(self.targets)
        #
        # return dset_orig

    def get_max_N_per_class(self, N, indexes=None, targets=None, seed=17):

        # Get targets from dset and indexes from its length
        if targets is None:
            targets = self.targets
        if indexes is None:
            indexes = range(0, self.length)

        # Constrain random generation
        np.random.seed(seed)

        # Extract indexes and corresponding classes
        np_targets = np.array(targets)
        unique_targets = np.unique(np_targets)
        valid_idx = []
        for t in unique_targets:
            pos = np.where(np_targets==t)[0]
            if len(pos) > N:
               pos = np.random.choice(pos, N, replace=False)
            valid_idx.extend(pos.tolist())
        return itemgetter(*valid_idx)(indexes), itemgetter(*valid_idx)(targets)

    def get_item_from_index(self, index):
        return self.images[index], self.targets[index]

    def split(self, ratios, save_load=True, **kwargs):
        pass

    def __getitem__(self, index):
        return self.get_item_from_index(index)

    def __len__(self):
        return self.length

    def __add__(self, dset):
        return self.append_subset(dset, create_new_dset=True)

    def __sub__(self, other):
        return self.diff_subset(other, create_new_dset=True)



