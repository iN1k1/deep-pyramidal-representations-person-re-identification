import itertools

def matirxscore2pairscore(matrix_score, idx_a, idx_b, id_a, id_b):
    pair_score = []
    pair_indexes = list(itertools.product(idx_a, idx_b))
    pair_ids = list(itertools.product(id_a, id_b))
    for ii in range(0, len(pair_indexes)):
        pair = pair_indexes[ii]
        pa = idx_a.index(pair[0])
        pb = idx_b.index(pair[1])
        pair_score.append(matrix_score[pa, pb])
    return pair_score, pair_indexes, pair_ids
