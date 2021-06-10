import numpy as np
import observations as obs
from covariance import CovarianceEstimator
import visualization as viz


results = dict(
    predictions = list(),
    pred_std = list(),
    observations = list()
)
# Transformation matrix
#   [x,dx, 0, 0]
#   [0,dx, 0, 0]
#   [0, 0, y,dy]
#   [0, 0, 0,dy]
F = np.matrix([
    [1., 1., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 1., 1.],
    [0., 0., 0., 1.],
])
# Observation matrix
H = np.identity(F.shape[0], dtype=F.dtype)
#   [x, 0, 0, 0]
#   [0,dx, 0, 0]
#   [0, 0, y, 0]
#   [0, 0, 0,dy]
I = np.identity(F.shape[0], dtype=F.dtype)


def main():

    # assumptions
    # Initial belief
    x0 = 0.0
    dx0 = 0.0
    y0 = 0.0
    dy0 = 0.0
    # Confidence in initial belief
    x_var0 = 100.0
    dx_var0 = 100.0
    y_var0 = 100.0
    dy_var0 = 100.0
    # Measurement noise
    loc_measErr = 1.5
    v_measErr = 0.25
    
    o = obs.Observations2d(start_x=4.0, start_y=6.0, decay_x=0.975, decay_y=0.985,
        noise_loc=loc_measErr, noise_vel=v_measErr
    )
    covar = CovarianceEstimator(n_dim=F.shape[0], dtype=F.dtype)
    x_t0 = np.matrix([x0, dx0, y0, dy0]).T
    R = np.matrix(np.diag(np.array(
        [50*loc_measErr**2, 50*v_measErr**2, 50*loc_measErr**2, 50*v_measErr**2]
    )))
    P = np.matrix(np.diag(np.array([x_var0, dx_var0, y_var0, dy_var0])))

    for _ in range(100):

        # prediction step
        x_t1 = F * x_t0

        # update
        # - error
        z = np.matrix(o.get()).T
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
        P = F*P*F.T + Q

        results['predictions'].append([
            float(x_t0[0,0]), float(x_t0[1,0]), float(x_t0[2,0]), float(x_t0[3,0])
        ])
        results['pred_std'].append([
            float(np.sqrt(Q[0,0])), float(np.sqrt(Q[1,1])), float(np.sqrt(Q[2,2])), float(np.sqrt(Q[3,3]))
        ])
        results['observations'].append([
            float(z[0,0]), float(z[1,0]), float(z[2,0]), float(z[3,0])
        ])
    
    viz.plot_2d_results(**results, file_name="kalman_visualization2d.png")

if __name__ == "__main__":
    main()