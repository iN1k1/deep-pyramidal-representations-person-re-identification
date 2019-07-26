from . import StandardMetric
import pickle
from . import dissimilarity


class DistanceMetric(object):
    def __init__(self, algorithm='euclidean', *args, **kwargs):
        super(DistanceMetric, self).__init__()
        self.algorithm = algorithm
        self.metric = StandardMetric()

    def __del__(self):
        pass

    def transform(self, X):
        return self.metric.transform(X)

    def pairwise_distance(self, Xa, Xb):
        Xa = self.transform(Xa)
        Xb = self.transform(Xb)
        method = self.algorithm
        return dissimilarity.pairwise(Xa, Xb, method=method, M=self.metric.M_)

    def save(self, path):
        with open(path + '.metric', 'wb') as f:
            pickle.dump(self.metric.M_, f)

    def load(self, path):
        with open(path + '.metric', 'rb') as f:
            self.metric.M_ = pickle.load(f)

