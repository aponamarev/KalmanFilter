import numpy as np
from observations import Observations
from covariance import CovarianceEstimator


# Transformation matrix
#   [x, velocity]
#   [0, velocity]
F = np.matrix([
    [1., 1.],
    [0., 1.]
])
H = np.identity(F.shape[0], dtype=F.dtype)
I = np.identity(F.shape[0], dtype=F.dtype)


def main():

    # assumptions
    x0 = 0.0
    v0 = 0.0
    x_measErr = 0.5
    v_measErr = 0.25
    x_var0 = 100.0
    v_var0 = 100.0
    
    o = Observations()
    covar = CovarianceEstimator(n_dim=F.shape[0], dtype=F.dtype)
    x_t0 = np.matrix([x0, v0]).T
    R = np.matrix(np.diag(np.array([x_measErr, v_measErr])))
    P = np.matrix(np.diag(np.array([x_var0, v_var0])))

    for _ in range(100):

        # prediction step
        x_t1 = F * x_t0

        # update
        z = np.matrix(o.get()).T
        y = z - H*x_t1
        # K = P / (P+R)
        S = H*P*H.T + R
        K = P*H.T * np.linalg.inv(S)
        # x[t] = x[t|t-1] * K*y
        x_t0 = x_t1 + K*y
        # P = (1-K)*P
        P = (I - K*H)*P
        # P = P + Q
        Q = covar.eval(y)
        P = H*P*H.T + Q
        print(
            f"Observation (z) - {z[0,0]:.2f}, "\
            f"prediction (x) - {x_t0[0,0]:.2f}, "\
            f"P[x] - {P[0,0]:.2f}, P[v] - {P[1,1]:.2f}, "\
            f"K[x] - {K[0,0]:.2f}, K[v] - {K[1,1]:.2f}"
        )

if __name__ == "__main__":
    main()