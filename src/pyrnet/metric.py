import time
from src.ml.metric.metric import DistanceMetric
import numpy as np
from src.reid.reranking.k_reciprocal import re_ranking

def get_distance(dset, X, metric, probe=None, gallery=None, multi_query=False, re_rank=False):

    t = time.time()
    print(' ==> Computing Dissimilarities...', end='')

    # Init metric
    metric = DistanceMetric(metric)

    # Do we have probe/gallery info?
    probe_mq = None
    if probe is None and gallery is None:
        if hasattr(dset, 'probe') and hasattr(dset, 'gallery') and len(dset.probe) > 0 and len(dset.gallery) > 0:
            probe = dset.probe
            gallery = dset.gallery

            # If available do we have to run a multi-query test?
            if isinstance(probe[0], list) or isinstance(probe[0], tuple):
                if multi_query:
                    probe_mq = probe[1]
                probe = probe[0]

        else:
            # All-vs-all
            probe = dset.indexes
            gallery = dset.indexes

    # Get probe and gallery indexes relative to the test set parition only
    _, probe_id, probe_idx, probe_cam = dset.get_items_from_indexes(probe)
    _, gallery_id, gallery_idx, gallery_cam = dset.get_items_from_indexes(gallery)

    # Multi-query info
    if multi_query and probe_mq is not None:
        _, probe_id_mq, probe_idx_mq, probe_cam_mq = dset.get_items_from_indexes(probe_mq)

    # Get probe and gallery features
    D = []
    D_rerank = []

    if multi_query and probe_mq is not None:
        probe_id_mq = np.array(probe_id_mq)
        probe_cam_mq = np.array(probe_cam_mq)
        probe_mq = np.array(probe_mq)
        for x in X:
            x_gallery = x[gallery]
            x_probe_m = np.zeros((len(probe), x.shape[1]), dtype=x_gallery.dtype)
            d = np.zeros((len(probe),len(gallery)))
            for ii in range(len(probe)):
                mquery_index = probe_mq[np.where((probe_id_mq == probe_id[ii]) & (probe_cam_mq == probe_cam[ii]))] # get sample by ID/cam
                x_probe_m[ii] = np.mean(x[mquery_index, :], axis=0)
                d[ii] = metric.pairwise_distance(x_probe_m[ii,:].reshape((1,-1)), x_gallery)

            # Re rank?
            if re_rank:
                D_rerank.append(_re_rank(metric, d, x_probe_m, x_gallery))

            # Keep track of dists
            D.append(d)

    else:
        for x in X:
            x_probe = x[probe]
            x_gallery = x[gallery]

            # Match images
            d = metric.pairwise_distance(x_probe, x_gallery)

            # Re rank?
            if re_rank:
                D_rerank.append(_re_rank(metric, d, x_probe, x_gallery))

            D.append(d)

    # Fuse dissimilarities
    dist = _fuse_dissimilarities(D)
    dist_rerank = None
    if re_rank:
        dist_rerank = _fuse_dissimilarities(D_rerank)

    print('done in {}'.format(time.time() - t))

    #
    return dist, dist_rerank, (probe, probe_id, probe_cam), (gallery, gallery_id, gallery_cam)


def _fuse_dissimilarities(dist, fuse_op='sum'):
    dist = np.array(dist)

    # Sum / max / prod?
    if fuse_op == 'sum':
        dist = np.sum(dist, axis=0)
    elif fuse_op == 'min':
        dist = np.min(dist, axis=0)
    elif fuse_op == 'prod':
        dist = np.prod(dist, axis=0)

    return dist

def _re_rank(metric, d, x_probe, x_gallery):
    q_q_dist = metric.pairwise_distance(x_probe, x_probe)
    g_g_dist = metric.pairwise_distance(x_gallery, x_gallery)
    return re_ranking(-d, -q_q_dist, -g_g_dist)
