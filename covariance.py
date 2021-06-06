import numpy as np


class CovarianceEstimator(object):

    def __init__(self, n_dim: int, dtype: type = np.float):

        self.c: np.matrix = np.matrix(np.zeros((n_dim, n_dim), dtype=dtype))
        self.n: int = 0
    
    def eval(self, values: np.matrix) -> np.matrix:
        assert isinstance(values, np.matrix)

        if values.shape[0] > 1:
            values = values.T
        
        self.c = np.concatenate((self.c, values), axis=0)
        self.n += 1

        delta = self.c - self.c.mean(axis=0)

        cov = 1/self.n * delta.T * delta

        return cov