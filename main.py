from States.Path import Path
from matplotlib import pyplot as plt
from Directories import df_sim_dir
import pandas as pd
from Graphs.save_fig import save_fig


def create_bar_chart(df, ax, block=False,):
    for filename, ts in zip(df['filename'], df['time series']):
        p = Path(time_step=0.25, time_series=ts)
        print(filename)
        p.bar_chart(ax=ax, axis_label=filename, block=block)
        if not block:
            ax.set_xlabel('time [min]')
        else:
            ax.set_xlabel('')


if __name__ == '__main__':
    time_series_dict, state_series_dict = Path.get_dicts(name='_selected_states_sim')
    df = pd.read_json(df_sim_dir)
    df['time series'] = df['filename'].map(time_series_dict)

    fig, ax = plt.subplots()
    block = False
    create_bar_chart(df, ax, block=block)
    save_fig(fig, 'pheidole')
    DEBUG = 1