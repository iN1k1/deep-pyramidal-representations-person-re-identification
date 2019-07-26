import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sklearn.metrics
import copy
from operator import itemgetter
#from .reid import ReIDPerformance

#
# def CMC_pair(pair_indexes, pair_ids, score, is_dist=False, method='new'):
#     """
#     Computes the Cumulative Matching Characteristic and
#     the corresponding normalized Area Under Curve
#
#     :param pair_indexes:
#         List of tuples defined as (probe_index, gallery_index)
#     :param pair_ids:
#         Lis of tuples defined as (probe_id, gallery_id)
#     :param score:
#         Similarity values between each pair in pair_indexes
#     :param is_dist:
#         True indicates that the score is a distance, not a similarity value
#     :return:
#         CMC curve
#         nAUC value
#         matching_indexes and matching_ids
#     """
#     idx = np.array(pair_indexes)
#     ids = np.array(pair_ids)
#     probe_idx, pos = np.unique(idx[:, 0], return_index=True)
#     probe_ids = ids[pos, 0]
#     gallery_idx = np.unique(idx[:, 1])
#     cmc = np.zeros((len(probe_idx), len(gallery_idx)))
#     cmc_old = np.zeros(len(gallery_idx))
#     score = np.array(score)
#     if is_dist:
#         score = -score
#     matching_indexes = []
#     matching_ids = []
#     average_precision = np.zeros(len(probe_idx))
#     #gallery_idx = np.unique(indexes[:, 1])
#
#     # Loop over probes
#     for ii, pidx in enumerate(probe_idx):
#
#         # Matching galleries
#         probe_gallery_pair_idx = np.where(idx[:,0]==pidx)[0]
#
#         # True match position?
#         # 1- Sort scores
#         sorted_idx = np.argsort(score[probe_gallery_pair_idx])[::-1] # Argsort sorts from small->large, we need the opposite
#         if not is_dist:
#             sorted_idx = sorted_idx[::-1]
#         matching_indexes.append( (pidx.tolist(), idx[probe_gallery_pair_idx[sorted_idx],1].tolist()) )
#
#         # 2- Check pos of true matches
#         sorted_gallery_ids = ids[probe_gallery_pair_idx[sorted_idx], 1]
#         match = np.where(sorted_gallery_ids == probe_ids[ii])[0]
#         matching_ids.append((probe_ids[ii].tolist(), sorted_gallery_ids.tolist()))
#
#         # 3- Increment Matches
#         cmc[ii, match[0]:] = 1
#         cmc_old[match] = cmc_old[match] + 1
#
#         # 4- Average precision
#         true_match = ids[probe_gallery_pair_idx,0]==ids[probe_gallery_pair_idx,1]
#         average_precision[ii] = sklearn.metrics.average_precision_score(true_match, score[probe_gallery_pair_idx])
#
#     # CMC
#     cmc = 100 * cmc.sum(axis=0) / len(probe_idx)
#     if method != 'new':
#         cmc = 100 * cmc_old.cumsum() / cmc_old.sum()
#
#     # Compute the nAUC
#     nauc = nAUC(cmc)
#
#     return cmc, nauc, average_precision, matching_indexes, matching_ids
#
#
# def reid_performance_from_mat(score_matrix, probe_idx, gallery_idx, probe_id, gallery_id, probe_cam=None, gallery_cam=None, same_cam=False, is_dist=False):
#     """
#     Computes the ReID performances
#     """
#
#     reid_perf = ReIDPerformance()
#     reid_perf.compute(score_matrix,
#                       probe_idx=probe_idx, gallery_idx=gallery_idx,
#                       probe_id=probe_id, gallery_id=gallery_id,
#                       probe_cam=probe_cam, gallery_cam=gallery_cam,
#                       same_cam=same_cam, is_dist=is_dist)

    # # Convert all indexes to numpy
    # probe_idx = np.array(probe_idx)
    # probe_id = np.array(probe_id)
    # gallery_idx = np.array(gallery_idx)
    # gallery_id = np.array(gallery_id)
    # if probe_cam is not None:
    #     probe_cam = np.array(probe_cam)
    # if gallery_cam is not None:
    #     gallery_cam = np.array(gallery_cam)
    #
    # cmc = np.zeros((len(probe_idx), len(gallery_idx)))
    # cmc_old = np.zeros(len(gallery_idx))
    # if is_dist:
    #     score_matrix = -score_matrix
    # matching_indexes = [[]] * len(probe_idx)
    # matching_ids = [[]] * len(probe_idx)
    # average_precision = np.zeros(len(probe_idx))
    # # gallery_idx = np.unique(indexes[:, 1])
    #
    # # Loop over probes
    # for ii, pidx in enumerate(probe_idx):
    #
    #     # Consider probe/gallery from same camera?
    #     valid_indexes = np.array(range(0, len(gallery_idx)))
    #     if not same_cam and probe_cam is not None and gallery_cam is not None:
    #         junk_indexes = np.where( (gallery_id == probe_id[ii]) & (gallery_cam == probe_cam[ii]) )[0]
    #         valid_indexes = np.setdiff1d(valid_indexes, junk_indexes)
    #
    #     # True match position?
    #     # 1- Sort scores
    #     sorted_idx = np.argsort(score_matrix[ii,valid_indexes])[::-1]  # Argsort sorts from small->large, we need the opposite
    #     sorted_idx = valid_indexes[sorted_idx]
    #     if not is_dist:
    #         sorted_idx = sorted_idx[::-1]
    #
    #     # Store list of matching indexes
    #     #matching_indexes.append((pidx.tolist(), gallery_idx[sorted_idx].tolist()))
    #     matching_indexes[ii] = (pidx.tolist(), gallery_idx[sorted_idx].tolist())
    #
    #     # 2- Check pos of true matches
    #     sorted_gallery_ids = gallery_id[sorted_idx]
    #     match = np.where(sorted_gallery_ids == probe_id[ii])[0]
    #     #matching_ids.append((probe_id[ii].tolist(), sorted_gallery_ids.tolist()))
    #     matching_ids[ii] = (probe_id[ii].tolist(), sorted_gallery_ids.tolist())
    #
    #     # 3- Increment Matches
    #     cmc[ii, match[0]:] = 1
    #     cmc_old[match] = cmc_old[match] + 1
    #
    #     # 4- Average precision
    #     true_match = gallery_id[valid_indexes] == probe_id[ii]
    #     if not same_cam and probe_cam is not None and gallery_cam is not None:
    #         true_match &= (gallery_cam[valid_indexes] != probe_cam[ii])
    #     average_precision[ii] = sklearn.metrics.average_precision_score(true_match, score_matrix[ii,valid_indexes])
    #
    # # CMC
    # cmc = 100 * cmc.sum(axis=0) / len(probe_idx)
    # if method != 'new':
    #     cmc = 100 * cmc_old.cumsum() / cmc_old.sum()
    #
    # # Compute the nAUC
    # nauc = nAUC(cmc)
    #
    # return cmc, nauc, average_precision, matching_indexes, matching_ids


def nAUC(curve):
    """
    Computes the normalized Area Under Curve
    :param curve: list of curve values
    :return: area under curve in [0,1]
    """
    return np.trapz(curve) / np.prod(curve.shape)


def precision_recall(actual, ranked_predictions, k=None, average=False):
    if not isinstance(actual, list):
        actual = [actual]
        ranked_predictions = [ranked_predictions]

    relevant = [[]] * len(actual)
    precision = [[]] * len(actual)
    recall = [[]] * len(actual)
    ap = [np.zeros(1)] * len(actual)
    for ii, act in enumerate(actual):
        if k is None:
            k = len(ranked_predictions[ii])

        # Top-k relevant
        relevant[ii] = ranked_predictions[ii][:k]==act

        # Precision and recall computation
        # To speed up the computation, we only consider relevant positions
        precision[ii] = np.zeros((k,))
        precision[ii][relevant[ii]] = 1
        precision[ii] = np.cumsum(precision[ii])

        # Copy precision
        recall[ii] = precision[ii].copy()

        # Compute precision by dividing it by number of considered item
        precision[ii] /= np.arange(1, k + 1)

        # Any relevant document in the top k?
        if relevant[ii].any():

            # Compute recall by dividing by the total number of relevant items
            recall[ii] /= np.cumsum(relevant[ii])

            # Fix NaNs by setting them to zero!
            recall[ii] = np.nan_to_num(recall[ii])

            # Compute average precision if any relevant document has been retrieved, otherwise it is zero!
            ap[ii] = np.sum(precision[ii] * relevant[ii]) / np.sum(relevant[ii])

        else:

            # Recall need to be set to zero!
            recall[ii] = np.zeros((k,))

    if average:
        precision = np.array(precision).mean(axis=0)
        recall = np.array(recall).mean(axis=0)
        ap = np.array(ap).mean()

    return precision, recall, ap, relevant


def average_precision(actual, predicted, k=None):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between an item and the retrieved list of items.
    Parameters
    ----------
    actual : int
             The class of the element that is searched
    predicted : list
                A list of predicted elements (sorted from best match to worst match!) => order does matter!
    k : int, optional
        The maximum number of elements to retrieve
    Returns
    -------
    score : double
            The average precision at k over the retrieved list
    """

    # Get precision, recall and index of relevant elements in the list
    precision, recall, ap, relevant = precision_recall(actual, predicted, k)

    # Compute the average precision only if there are relevant documents retrieved in the top-k ones!
    # Otherwise, precision is 0
    #p = [[0]] * len(relevant)
    #for ii, r in enumerate(relevant):
    #    if len(r) > 0:
    #        p[ii] = precision[ii][r].mean()

    return ap


def mean_average_precision(actual, predicted, k=None):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([average_precision(a,p,k) for a,p in zip(actual, predicted)])


def multi_label_performance(actual, predicted):

    # Number of classes
    n_classes = actual.shape[1]

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = sklearn.metrics.precision_recall_curve(actual[:, i], predicted[:, i])
        average_precision[i] = sklearn.metrics.average_precision_score(actual[:, i], predicted[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = sklearn.metrics.precision_recall_curve(actual.ravel(),
                                                                                    predicted.ravel())
    average_precision["micro"] = sklearn.metrics.average_precision_score(actual, predicted, average="micro")

    return precision, recall, average_precision


def get_matching_images(probe_dataset, gallery_dataset, matching_indexes, N, selected_indexes=None):
    matching_images = []
    if selected_indexes is None:
        selected_indexes = range(len(matching_indexes))

    # Selected indexes
    selected_matching_indexes = itemgetter(*selected_indexes)(matching_indexes)

    for ii in range(0, len(selected_matching_indexes)):
        pidx = selected_matching_indexes[ii][0]
        gidx = selected_matching_indexes[ii][1][:N]
        imp = probe_dataset[pidx][0]
        img = []
        for g in gidx:
            img.append(gallery_dataset[g][0])

        matching_images.append((imp, img))

    return matching_images
