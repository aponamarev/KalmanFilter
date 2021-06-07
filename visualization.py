from matplotlib import pyplot as plt


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



