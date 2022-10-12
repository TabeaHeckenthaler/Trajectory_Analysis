from States.Path import Path
from matplotlib import pyplot as plt
from Directories import df_sim_dir, project_home
import pandas as pd
from os import path, mkdir
import datetime


def create_bar_chart(df, ax, block=False,):
    for filename, ts in zip(df['filename'], df['time series']):
        p = Path(time_step=0.25, time_series=ts)
        print(filename)
        p.bar_chart(ax=ax, axis_label=filename, block=block)
        if not block:
            ax.set_xlabel('time [min]')
        else:
            ax.set_xlabel('')


def graph_dir():
    direct = path.abspath(path.join(project_home, 'Graphs',
                                    datetime.datetime.now().strftime("%Y") + '_' +
                                    datetime.datetime.now().strftime("%m") + '_' +
                                    datetime.datetime.now().strftime("%d")))
    if not (path.isdir(direct)):
        mkdir(direct)
    return direct


def save_fig(fig, name):
    name = "".join(x for x in name if x.isalnum())
    if fig.__module__ == 'plotly.graph_objs._figure':
        fig.write_image(graph_dir() + path.sep + name + '.pdf')
        fig.write_image(graph_dir() + path.sep + name + '.svg')
    else:
        fig.savefig(graph_dir() + path.sep + name + '.pdf', format='pdf', pad_inches=0.5, bbox_inches='tight')
        fig.savefig(graph_dir() + path.sep + name + '.svg', format='svg', pad_inches=0.5, bbox_inches='tight')


if __name__ == '__main__':
    time_series_dict, state_series_dict = Path.get_dicts(name='_selected_states_sim')
    df = pd.read_json(df_sim_dir)
    df['time series'] = df['filename'].map(time_series_dict)

    fig, ax = plt.subplots()
    block = False
    create_bar_chart(df, ax, block=block)
    save_fig(fig, 'pheidole')
    DEBUG = 1