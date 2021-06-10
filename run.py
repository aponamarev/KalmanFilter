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
H = np.matrix([1.0, 0.0], dtype=F.dtype)
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
    x_measErr = 5.0
    
    o = Observations(start=100.0, decay=0.99, noise_x=x_measErr)
    covar = CovarianceEstimator(n_dim=H.shape[0], dtype=F.dtype)
    x_t0 = np.matrix([x0, v0]).T
    R = np.matrix(np.diag(np.array([50*x_measErr**2, ])))
    P = np.matrix(np.diag(np.array([x_var0, v_var0])))

    for _ in range(100):

        # prediction step
        x_t1 = F * x_t0

        # update
        # - error
        z = np.matrix([o.get()[0],])
        y = z - H*x_t1

        # - Kalman gain
        # K = P / (P+R)
        S = H*P*H.T + R
        K = P*H.T * np.linalg.inv(S)

        # - state update
        # x[t] = x[t|t-1] * K*y
        x_t0 = x_t1 + K*y

        # - prediction error update
        # P = (1-K)*P
        P = (I - K*H)*P
        # P = P + Q
        Q = covar.eval(y)
        P = F*P*F.T + H.T*Q*H

        results['predictions'].append(float(x_t0[0,0]))
        results['pred_std'].append(float(np.sqrt(Q[0,0])))
        results['observations'].append(float(z[0,0]))
    
    plot_results(**results, file_name="kalman_visualization.png")

if __name__ == "__main__":
    main()