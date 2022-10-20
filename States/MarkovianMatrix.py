from States.Path import Path
from States.Network import Network, state_order
import pathpy
from main import df_chosen
from matplotlib import pyplot as plt
import numpy as np
from itertools import groupby
import pandas as pd

colors = {'XL': 'red', 'L': 'blue', 'M': 'k', 'S': 'green'}


def plot_T(T: pd.DataFrame):
    fig, axis = plt.subplots(1, 1)
    to_plot = T.reindex(state_order)[state_order]
    to_plot = np.log(to_plot)
    _ = axis.imshow(to_plot)
    axis.set_xticks(range(len(to_plot)))
    axis.set_xticklabels(to_plot.columns, fontsize=5)
    axis.set_yticks(range(len(to_plot)))
    axis.set_yticklabels(to_plot.columns, fontsize=7)
    plt.savefig('transition_matrix_' + str(transition_boolean) + size)


def plot_t(t: pd.Series, ax, error=None, linestyle='solid', color=None, label=None):
    to_plot = t.reindex(state_order)[state_order]
    if error is not None:
        error = error.reindex(state_order)[state_order]
    _ = ax.errorbar(x=range(len(to_plot)), y=to_plot,  yerr=error, label=label, color=color, linestyle=linestyle)
    ax.set_xticks(range(len(to_plot)))
    ax.set_xticklabels(to_plot.index, fontsize=8)


# define the needed experiments
solver = 'ant'
sizes = ['XL', 'L', 'M', 'S']

fig_solving_time, axis = plt.subplots(1, 2)


for size in sizes:
    shape = 'SPT'

    # Amir: You need here a list of file names!
    df_chosen_size = df_chosen[df_chosen['size'] == size]
    filenames = df_chosen_size['filename']

    # Amir: Here you load the time_series of the list of file names
    time_series_dict, state_series_dict = Path.get_dicts(name='_selected_states')
    time_series_dict_new = {filename: time_series_dict[filename] for filename in filenames}

    # calculate the Markovian Matrix
    for transition_boolean, ax in zip([False, True], axis):
        paths = pathpy.Paths()
        paths.max_subpath_length = 2
        real_paths = []
        for p in time_series_dict_new.values():
            if transition_boolean:
                p = [''.join(ii[0]) for ii in groupby([tuple(label) for label in p])]
            paths.add_path(p, expand_subpaths=True)
            real_paths.append(p)

        n = Network.init_from_paths(paths, solver, shape, size=size)
        T, t = n.markovian_analysis()
        t_real = n.absorption_time(real_paths)
        t_real = pd.DataFrame(t_real, index=['mean', 'sem']).transpose()

        plot_t(t, ax, linestyle='dashed', color=colors[size], label=size + ' markovian')
        plot_t(t_real['mean'], ax, error=t_real['sem'], color=colors[size], label=size + ' experimental')
        # plot_T(T)

axis[0].set_title('ants: time series')
axis[1].set_title('ants: state series')
plt.legend()
plt.savefig('solving_time_', dpi=150)
DEBUG = 1
