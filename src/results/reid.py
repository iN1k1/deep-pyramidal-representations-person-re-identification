import sklearn.metrics
import numpy as np
from .performance import nAUC


class ReIDPerformance(object):
    def __init__(self):
        super(ReIDPerformance, self).__init__()

        self.cmc = None
        self.ap = None
        self.nauc = None
        self.matching_indexes = None
        self.matching_ids = None
        self.probes_performance = None

        self.probe_idx = None
        self.gallery_idx = None
        self.probe_id = None
        self.gallery_id = None
        self.probe_cam = None
        self.gallery_cam = None

        self.score_matrix = None

    def compute(self, score_matrix, probe_idx, gallery_idx, probe_id, gallery_id,
                probe_cam=None, gallery_cam=None, pool_over_cam_operator=''):

        # Convert all indexes to numpy
        self.probe_idx = np.array(probe_idx)
        self.probe_id = np.array(probe_id)
        self.gallery_idx = np.array(gallery_idx)
        self.gallery_id = np.array(gallery_id)
        if probe_cam is not None:
            self.probe_cam = np.array(probe_cam)
        if gallery_cam is not None:
            self.gallery_cam = np.array(gallery_cam)

        # Local copy
        self.score_matrix = score_matrix

        # Init list of probe performances
        self.probes_performance = [[] for _ in range(len(probe_idx))]

        # Loop over probes
        for ii in range(len(probe_idx)):
            self.probes_performance[ii] = self.compute_probe_performance(score_matrix[ii], self.probe_id[ii], self.gallery_id,
                                                                         self.probe_idx[ii], self.gallery_idx,
                                                                         self.probe_cam[ii], self.gallery_cam, pool_over_cam_operator)

        # Update overall performances

        # Average precision
        self.ap = np.array([perf.ap for perf in self.probes_performance])

        # CMC
        self.cmc = np.vstack([perf.cmc for perf in self.probes_performance])
        self.cmc = 100 * self.cmc.sum(axis=0) / len(probe_idx)

        # nAUC
        self.nauc = nAUC(self.cmc)

        # Matching IDs
        self.matching_ids = [perf.matching_ids for perf in self.probes_performance]

        # Matching indexes
        self.matching_indexes = [perf.matching_indexes for perf in self.probes_performance]

    def compute_probe_performance(self, score, probe_id, gallery_id, probe_idx, gallery_idx, probe_cam, gallery_cam, pool_over_cam_operator=''):
        probe_performance = ProbePerformance()
        probe_performance.compute(score, probe_idx=probe_idx, gallery_idx=gallery_idx,
                                  probe_id=probe_id, gallery_id=gallery_id,
                                  probe_cam=probe_cam, gallery_cam=gallery_cam, pool_over_cam_operator=pool_over_cam_operator)
        return probe_performance


class ProbePerformance(object):
    def __init__(self, probe_id=None, probe_idx=None, probe_cam=None):
        super(ProbePerformance, self).__init__()

        self.probe_id = probe_id
        self.probe_idx = probe_idx
        self.probe_cam = probe_cam
        self.cmc = None
        self.ap = None
        self.matching_indexes = None
        self.matching_ids = None

    def compute(self, score, gallery_idx, gallery_id,
                probe_idx=None, probe_id=None, probe_cam=None, gallery_cam=None, same_cam=False, pool_over_cam_operator=''):

        # Update probe info if necessary
        if probe_id is not None:
            self.probe_id = probe_id
        if probe_idx is not None:
            self.probe_idx = probe_idx
        if probe_cam is not None:
            self.probe_cam = probe_cam

        # Consider probe/gallery from same camera?
        valid_indexes = np.array(range(len(gallery_idx)))
        if not same_cam and self.probe_cam is not None and gallery_cam is not None:
            junk_indexes = np.where((gallery_id == probe_id) & (gallery_cam == self.probe_cam))[0]
            valid_indexes = np.setdiff1d(valid_indexes, junk_indexes)

        # Pool over camera?
        if self.probe_cam is not None and gallery_cam is not None and pool_over_cam_operator != '':
            pooled_score = []
            pooled_gallery_id = []
            pooled_gallery_cam = []
            pooled_gallery_idx = []
            pool_operator = np.max
            if pool_over_cam_operator == 'avg' or pool_over_cam_operator == 'mean':
                pool_operator = np.mean
            if pool_over_cam_operator == 'min':
                pool_operator = np.min
            for cam in np.unique(gallery_cam):
                for pid in np.unique(gallery_id):
                    idx = np.where((cam == gallery_cam) & (pid == gallery_id))
                    gallery_idx_cam = np.intersect1d(valid_indexes, idx)
                    if len(gallery_idx_cam) > 0:
                        pooled_score.append(pool_operator(score[gallery_idx_cam]))
                        pooled_gallery_idx.append(gallery_idx_cam[0])
                        pooled_gallery_id.append(pid)
                        pooled_gallery_cam.append(cam)
            score = np.array(pooled_score)
            gallery_id = np.array(pooled_gallery_id)
            gallery_idx = np.array(pooled_gallery_idx)
            gallery_cam = np.array(pooled_gallery_cam)
            valid_indexes = np.array(range(len(score)))


        # True match position?
        # 1- Sort scores
        sorted_idx = np.argsort(score[valid_indexes])[::-1]  # Argsort sorts from small->large, we need the opposite
        sorted_idx = valid_indexes[sorted_idx]

        # Store list of matching indexes
        # matching_indexes.append((pidx.tolist(), gallery_idx[sorted_idx].tolist()))
        self.matching_indexes = (self.probe_idx.tolist(), gallery_idx[sorted_idx].tolist())

        # 2- Check pos of true matches
        sorted_gallery_ids = gallery_id[sorted_idx]
        match = np.where(sorted_gallery_ids == probe_id)[0]
        # matching_ids.append((probe_id[ii].tolist(), sorted_gallery_ids.tolist()))
        self.matching_ids = (self.probe_id.tolist(), sorted_gallery_ids.tolist())

        # 3- CMC
        self.cmc = np.zeros((len(gallery_idx), ))
        self.cmc[match[0]:] = 1
        # cmc_old[match] = cmc_old[match] + 1

        # 4- Average precision
        true_match = gallery_id[valid_indexes] == probe_id
        if not same_cam and probe_cam is not None and gallery_cam is not None:
            true_match &= (gallery_cam[valid_indexes] != probe_cam)
        self.ap = sklearn.metrics.average_precision_score(true_match, score[valid_indexes])
