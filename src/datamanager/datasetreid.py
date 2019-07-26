from .dataset import Dataset
from . import utils as datautils
from ..utils import misc
import math
import os
import numpy as np
import copy
from operator import itemgetter


class DatasetReID(Dataset):
    def __init__(self, name, root_folder, load=True, im_size=None, in_memory=False, keep_aspect_ratio=True):
        super(DatasetReID, self).__init__(name, root_folder, im_size=im_size)
        self.cams = []
        self.frames = []
        self.indexes = []
        self.probe = []
        self.gallery = []
        loaded_from_file = False
        if load:
            loaded_from_file = self.load(path=self.data_path, in_memory=in_memory, keep_aspect_ratio=keep_aspect_ratio)
            if len(self.images) == 0:
                raise (RuntimeError("Found 0 images in : " + root_folder))

            # If data has not been loaded from file, we can save it!
            if not loaded_from_file:
                self.save(self.data_path)

    def load(self, path=None, in_memory=False, keep_aspect_ratio=True):
        loaded_from_file = False
        if path is None:
            path = self.data_path
        if os.path.exists(path):
            data = misc.load(path)
            self.images, self.targets, self.cams, self.frames, self.indexes, \
            self.classes, self.class_to_idx, self.probe, self.gallery = data['images'], data['targets'], data['cams'], data['frames'], data['indexes'], \
                                                                        data['classes'], data['class_to_idx'], data['probe'], data['gallery']

            loaded_from_file = True
        else:
            self.images, self.targets, self.cams, self.frames, self.indexes = datautils.load_reid_dataset(self.root, in_memory, self.im_size, keep_aspect_ratio)

            # Make contiguous IDs
            self.targets, self.classes, self.class_to_idx = datautils.make_contiguous_targets(self.targets)

        self.compute_idx_to_class()
        self.length = len(self.images)
        return loaded_from_file

    def save(self, path=None):
        if path is None:
            path = self.data_path
        misc.save(path, images=self.images, targets=self.targets, cams=self.cams, frames=self.frames, indexes=self.indexes,
                  classes=self.classes, class_to_idx=self.class_to_idx, probe=self.probe, gallery=self.gallery)

    def get_indexes_from_cam(self, cam, N=None):
        indexes = [idx for idx in range(0, self.length) if self.cams[idx] == cam]
        targets = itemgetter(*indexes)(self.targets)
        cams = itemgetter(*indexes)(self.cams)
        if N is not None:
            indexes, targets, cams = self.get_max_N_per_class(N, indexes=indexes, targets=targets, cams=cams)
        return indexes, targets, cams

    def get_item_from_global_index(self, idx, N=None):
        indexes = [self.indexes.index(ii) for ii in idx]
        targets = itemgetter(*indexes)(self.targets)
        cams = itemgetter(*indexes)(self.cams)
        if N is not None:
            indexes, targets, cams = self.get_max_N_per_class(N, indexes=indexes, targets=targets, cams=cams)
        return indexes, targets, cams

    def get_indexes_from_ID(self, ID):
        return [idx for idx in range(0, self.length) if self.targets[idx] == ID]

    def get_indexes_from_id_cam(self, target, cam):
        indexes = [idx for idx in range(0, self.length) if self.cams[idx] == cam and self.targets[idx] == target]
        if len(indexes) == 0:
            return indexes, None, None
        targets = itemgetter(*indexes)(self.targets)
        cams = itemgetter(*indexes)(self.cams)
        return indexes, targets, cams

    def get_item_from_index(self, index):
        return self.images[index], self.targets[index], self.cams[index], self.indexes[index]

    def get_items_from_indexes(self, indexes):
        images = itemgetter(*indexes)(self.images)
        targets = itemgetter(*indexes)(self.targets)
        cams = itemgetter(*indexes)(self.cams)
        indexes = itemgetter(*indexes)(self.indexes)
        return images, targets, indexes, cams

    def get_index(self, target, cam, frame=None):
        index = [idx for idx in range(0, self.length) if self.cams[idx] == cam and self.targets[idx] == target and self.frames[idx] == frame][0]
        return self.get_item_from_index(index), index

    def get_max_N_per_class(self, N, indexes=None, targets=None, cams=None):

        # Get targets from dset and indexes from its length
        if targets is None:
            targets = self.targets
        if indexes is None:
            indexes = range(0, self.length)
        if cams is None:
            cams = self.cams

        # Extract indexes and corresponding classes
        np_targets = np.array(targets)
        unique_targets = np.unique(np_targets)
        valid_idx = []
        for t in unique_targets:
            pos = np.where(np_targets == t)[0]
            if len(pos) > N:
                pos = np.random.choice(pos, N, replace=False)
            valid_idx.extend(pos.tolist())
        return itemgetter(*valid_idx)(indexes), itemgetter(*valid_idx)(targets), itemgetter(*valid_idx)(cams)

    def split(self, ratios, save_load=True, **kwargs):

        dset_train = DatasetReID(self.name + '_train_' + str(ratios[0]), self.root, False, self.im_size, not isinstance(self.images[0], str))
        dset_test = DatasetReID(self.name + '_test_' + str(ratios[1]), self.root, False, self.im_size, not isinstance(self.images[0], str))

        # Try to load from data file
        if save_load and os.path.exists(dset_train.data_path) and os.path.exists(dset_test.data_path):
            dset_train.load()
            dset_test.load()
        else:
            probe_idx = []
            probe_idx_multi = []

            # Dataset is already partitioned..
            if os.path.exists(os.path.join(self.root, 'train')):
                n_train = len(os.listdir(os.path.join(self.root, 'train')))
                tridx = list(range(0, n_train))
                teidx = list(range(n_train, self.length))

                # Do we have query images already specifid?
                n_query = len(os.listdir(os.path.join(self.root, 'query')))
                n_query_m = 0
                if os.path.exists(os.path.join(self.root, 'query_multi')):
                    n_query_m = len(os.listdir(os.path.join(self.root, 'query_multi')))

                probe_idx = list(range(self.length - (n_query+n_query_m), self.length-n_query_m))
                probe_idx_multi = list(range(self.length - n_query_m, self.length))
            else:
                unique_targets = list(set(self.targets))
                if ratios[0] <= 1 and ratios[1] <= 1:
                    t = math.floor( len(unique_targets)*ratios[0] )
                    ratios = (t, len(unique_targets)-t)
                perm = np.random.permutation(unique_targets)
                tridx = list(np.sort([ii for ii in range(0,self.length) if self.targets[ii] in perm[:ratios[0]] ]).tolist())
                teidx = list(np.sort([ii for ii in range(0,self.length) if self.targets[ii] in perm[ratios[0]:ratios[0]+ratios[1]] ]).tolist())

            dset_train = self.extract_subset(tridx, dset=dset_train)
            dset_test = self.extract_subset(teidx, dset=dset_test)

            if 'make_each_split_contiguous' in kwargs and kwargs['make_each_split_contiguous']:
                dset_train.targets, dset_train.classes, dset_train.class_to_idx = datautils.make_contiguous_targets(dset_train.targets)
                dset_test.targets, dset_test.classes, dset_test.class_to_idx = datautils.make_contiguous_targets(dset_test.targets)

            # Are the prpbe (query) images already specified?
            if probe_idx != []:

                # Update probe and gallery according to the tridx/teidx split..
                dset_test.probe = [dset_test.indexes.index(ii) for ii in probe_idx]
                gallery_idx = list(set(dset_test.indexes) - set(probe_idx))
                dset_test.gallery = [dset_test.indexes.index(ii) for ii in gallery_idx]

            if probe_idx_multi != []:
                probe_multi = [dset_test.indexes.index(ii) for ii in probe_idx_multi]
                probe_idx_multi_all = copy.deepcopy(probe_idx)
                probe_idx_multi_all.extend(probe_idx_multi)
                gallery_idx = list(set(dset_test.indexes) - set(probe_idx_multi_all))
                dset_test.probe = [dset_test.probe, probe_multi]
                dset_test.gallery = [dset_test.indexes.index(ii) for ii in gallery_idx]

            # Save
            if save_load:
                dset_train.save()
                dset_test.save()

        return dset_train, dset_test

    def extract_subset(self, idx, dset=None):

        # Create clone
        if dset is None:
            dset = DatasetReID(self.name, self.root, im_size=self.im_size)

        dset.class_to_idx = copy.copy(self.class_to_idx)
        dset.idx_to_class = copy.copy(self.idx_to_class)
        dset.compute_idx_to_class()

        # Get only selected samples
        if len(idx) > 0:
            dset.images = itemgetter(*idx)(self.images)
            dset.targets = itemgetter(*idx)(self.targets)
            dset.cams = itemgetter(*idx)(self.cams)
            dset.indexes = itemgetter(*idx)(self.indexes)
            if isinstance(dset.targets, int):
                dset.targets = [dset.targets]
                dset.images = [dset.images]
                dset.cams = [dset.cams]
                dset.indexes = [dset.indexes]
            dset.classes = np.sort(np.unique(dset.targets))
            dset.length = len(dset.targets)

        return dset

    def get_paired_dataset(self, campair, targets_to_pos_neg=False, N=999999, only_positives=False):
        dset_pair = DatasetReID(self.name + '_paired', self.root, False, self.im_size, not isinstance(self.images[0], str))
        dset_pair.images, dset_pair.targets, dset_pair.cams, dset_pair.indexes = datautils.compute_pairs_cams(self.images, self.targets,
                                                                                                              self.cams,range(0,len(self)), campair, N)
        # From target pairs (id1, id2) to +1/-1
        if targets_to_pos_neg:
            dset_pair.targets = (np.array(dset_pair.targets)[:,0] == np.array(dset_pair.targets)[:,1]).tolist()

        # Remove negative pairs
        if only_positives:
            pos_idx = []
            if targets_to_pos_neg:
                pos_idx = [ii for ii, idx in enumerate(dset_pair.targets) if idx == True]
            else:
                pos_idx = [ii for ii, idx in enumerate(dset_pair.targets) if idx[0] == idx[1]]

            # Keep only positives!
            dset_pair.images = itemgetter(*pos_idx)(dset_pair.images)
            dset_pair.targets = itemgetter(*pos_idx)(dset_pair.targets)
            dset_pair.cams = itemgetter(*pos_idx)(dset_pair.cams)

            # Remap indexes..
            dset_pair.indexes = itemgetter(*pos_idx)(dset_pair.indexes)



        # Length of the dataset
        dset_pair.length = len(dset_pair.images)

        return dset_pair
