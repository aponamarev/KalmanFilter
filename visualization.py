from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d


def plot_results(predictions: list, pred_std: list, observations: list=None, file_name: str=None) -> None:

    assert len(predictions) == len(observations)

    x = list(range(len(predictions)))

    fig, ax = plt.subplots()

    l_bound, u_bound = list(zip(*[(p-c, p+c) for p, c in zip(predictions, pred_std)]))

    if observations is not None:
        ax.plot(x,observations, color='r', label='observations')
        
    ax.plot(x,predictions, color='b', label='predictions', linewidth=2)
    ax.fill_between(x, l_bound, u_bound, color='b', alpha=.1, label='conf')

    plt.legend(loc='lower right')

    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()


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



