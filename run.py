import numpy as np
from observations import Observations
from covariance import CovarianceEstimator
from visualization import plot_results


results = dict(
    predictions = list(),
    pred_std = list(),
    observations = list()
)
# Transformation matrix
#   [x, velocity]
#   [0, velocity]
F = np.matrix([
    [1., 1.],
    [0., 1.]
])
# Observation matrix
H = np.identity(F.shape[0], dtype=F.dtype)
#   [x, 0],
#   [., v]
I = np.identity(F.shape[0], dtype=F.dtype)


def main():

    # assumptions
    # Initial belief
    x0 = 0.0
    v0 = 0.0
    # Confidence in initial belief
    x_var0 = 100.0
    v_var0 = 100.0
    # Measurement noise
    x_measErr = 1.5
    v_measErr = 0.25
    
    o = Observations(decay=0.975, noise_x=x_measErr, noise_v=v_measErr)
    covar = CovarianceEstimator(n_dim=F.shape[0], dtype=F.dtype)
    x_t0 = np.matrix([x0, v0]).T
    R = np.matrix(np.diag(np.array([x_measErr, v_measErr])))
    P = np.matrix(np.diag(np.array([x_var0, v_var0])))

    for _ in range(100):

        # prediction step
        x_t1 = F * x_t0

        # update
        z = np.matrix(o.get()).T

        results['predictions'].append(float(x_t1[0,0]))
        results['pred_std'].append(float(np.sqrt(P[0,0])))
        results['observations'].append(float(z[0,0]))

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
        P = F*P*F.T + Q
    
    plot_results(**results, file_name="kalman_visualization.png")

if __name__ == "__main__":
    main()