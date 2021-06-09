from typing import Iterable
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d


def plot_results(predictions: list, pred_std: list, observations: list=None, k_gain: list=None, file_name: str=None) -> None:

    assert len(predictions) == len(observations)

    x = list(range(len(predictions)))
    p_n = 1 if not isinstance(predictions[0], Iterable) else len(predictions[0])
    labels = ["x_pred","v_pred","a_pred"]
    colors = ["blue", "green", "yellow"]
    if p_n == 1:
        predictions = [[p,] for p in predictions]

    fig, axs = plt.subplots() if p_n == 1 else plt.subplots(1, 2 if k_gain is None else 3, figsize=(12,6))

    l_bound, u_bound = list(zip(*[(p[0]-c, p[0]+c) for p, c in zip(predictions, pred_std)]))
    ax = axs if p_n == 1 else axs[0]
    ax.fill_between(x, l_bound, u_bound, color=colors[0], alpha=.1, label='+/- std')

    if observations is not None:
        o_n = 1 if not isinstance(observations[0], Iterable) else len(observations[0])
        ax = axs if p_n == 1 else axs[0]
        ax.plot(
            x,
            observations if o_n==1 else [o[0] for o in observations], 
            color='r', label='observations'
        )
    
    if k_gain is not None:
        k_n = 1 if not isinstance(k_gain[0], Iterable) else len(k_gain[0])
        ax = axs if p_n == 1 else axs[2]
        for i in range(k_n):
            g = [g[i] for g in k_gain]
            label = labels[i]
            color = colors[i]
            ax.plot(x, g, color=color, label=label, linewidth=2)

        
    for i in range(p_n):
        ax = axs if p_n == 1 else axs[min(i,1)]
        p = [p[i] for p in predictions]
        label = labels[i]
        color = colors[i]
        ax.plot(x, p, color=color, label=label, linewidth=2)

    fig.legend(loc='lower right')

    if file_name is not None:
        fig.savefig(file_name)
    else:
        fig.show()


def plot_2d_results(predictions: list, pred_std: list, observations: list=None, file_name: str=None) -> None:

    assert len(predictions) == len(observations)

    z = list(range(len(predictions)))

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    if observations is not None:
        x_obs, y_obs = list(zip(*[
            (float(observ[0]), float(observ[2])) for observ in observations
        ]))
        ax.scatter3D(z, x_obs, y_obs, c=z, cmap='Reds', label='observations')
    
    x_p, x_p_ub, x_p_lb, y_p, y_p_ub, y_p_lb = list(zip(*[
        (float(p[0]), float(p[0]+s[0]), float(p[0]-s[0]), float(p[2]), float(p[2]+s[0]), float(p[2]-s[0])) 
        for p, s in zip(predictions, pred_std)
    ]))
    ax.scatter3D(z, x_p, y_p, c=z, cmap='Blues', label='predict')
    ax.scatter3D(z, x_p_ub, y_p_ub, c=z, cmap='Greens', label='+ø')
    ax.scatter3D(z, x_p_lb, y_p_lb, c=z, cmap='Greens', label='-ø')

    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()



