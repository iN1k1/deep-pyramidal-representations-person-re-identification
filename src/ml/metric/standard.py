
class StandardMetric(object):
    def __init__(self):
        super(StandardMetric, self).__init__()
        self.M_ = None

    def metric(self):
        return None

    def fit(self, X):
        self.X_ = X

    def transform(self, X=None):
        if X is None:
            return self.X_
        return X

