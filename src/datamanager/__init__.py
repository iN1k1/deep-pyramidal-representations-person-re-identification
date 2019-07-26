import os
from .utils import load_image, copy_image
from .dataset import Dataset
from .datasetreid import DatasetReID
from .dataprovider import DataProvider
from .transformer import DataTransformer, ToNumpy, RandomErasing


def get_dataset(dset_opts, save=True, split_tr_te_perc=None):

    dset_root_folder = os.path.join('data', 'ReID', dset_opts.name)
    data_loader = load_image

    dset = DatasetReID(dset_opts.name, dset_root_folder)
    if split_tr_te_perc is not None:
        dset_train, dset_test = dset.split(ratios=split_tr_te_perc)
    else:
        dset_train = dset
        dset_test = []

    if save:
        try:
            dset_train.save()
            dset_test.save()
        except AttributeError as e:
            print('Unable to save dataset paritions')

    return dset_train, dset_test, data_loader

__all__ = ['Dataset', 'DatasetReID', 'DataProvider', 'DataTransformer', 'ToNumpy', 'get_dataset', 'RandomErasing']

