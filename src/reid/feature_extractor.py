import numpy as np
from operator import itemgetter
from sklearn import preprocessing


class FeatureExtractor(object):

    def __init__(self):
        super(FeatureExtractor, self).__init__()

    def extract_features(self, dset, opts, main_path='.'):
        pass

    def get_train_test_features(self, features, dsetTr, dsetTe, camPair, N):

        # Split over train/test features
        X_train = itemgetter(*dsetTr.indexes)(features)
        X_test = itemgetter(*dsetTe.indexes)(features)

        if camPair[0] == -1 or camPair[1] == -1:
            idx = np.random.permutation(dsetTr.indexes).tolist()
            idx_tr_a, id_tr_a, cam_tr_a = dsetTr.get_item_from_global_index(idx[0:int(len(idx)/2)], N=N)
            idx_tr_b, id_tr_b, cam_tr_b = dsetTr.get_item_from_global_index(idx[int(len(idx)/2):], N=N)
        else:
            idx_tr_a, id_tr_a, cam_tr_a = dsetTr.get_indexes_from_cam(camPair[0], N=N)
            idx_tr_b, id_tr_b, cam_tr_b = dsetTr.get_indexes_from_cam(camPair[1], N=N)

        if dsetTe.query != []:
            idx_te_a, id_te_a, cam_te_a = dsetTe.get_item_from_global_index(dsetTe.query, N=N)
            idx_te_b, id_te_b, cam_te_b = dsetTe.get_item_from_global_index(dsetTe.gallery, N=N)
        else:
            idx_te_a, id_te_a, cam_te_a = dsetTe.get_indexes_from_cam(camPair[0], N=N)
            idx_te_b, id_te_b, cam_te_b = dsetTe.get_indexes_from_cam(camPair[1], N=N)

        apply_normalizations = False
        X_train_a = self.list2np(itemgetter(*idx_tr_a)(X_train), apply_normalizations=apply_normalizations)
        X_train_b = self.list2np(itemgetter(*idx_tr_b)(X_train), apply_normalizations=apply_normalizations)

        X_test_a = self.list2np(itemgetter(*idx_te_a)(X_test), apply_normalizations=apply_normalizations)
        X_test_b = self.list2np(itemgetter(*idx_te_b)(X_test), apply_normalizations=apply_normalizations)

        # Standardize features
        X_all = preprocessing.scale(np.concatenate((X_train_a, X_train_b, X_test_a, X_test_b)))
        idx = np.array([X_train_a.shape[0], X_train_b.shape[0], X_test_a.shape[0], X_test_b.shape[0]])
        X_train_a = X_all[:idx[0]]
        X_train_b = X_all[idx[0]:idx[range(0,2)].sum()]
        X_test_a = X_all[idx[range(0,2)].sum():idx[range(0,3)].sum()]
        X_test_b = X_all[idx[range(0,3)].sum():]

        return X_train_a, X_train_b, X_test_a, X_test_b, \
               idx_tr_a, id_tr_a, idx_tr_b, id_tr_b, cam_tr_a, cam_tr_b, \
               idx_te_a, id_te_a, idx_te_b, id_te_b, cam_te_a, cam_te_b

    def list2np(self, featList, apply_normalizations=False):
        pass