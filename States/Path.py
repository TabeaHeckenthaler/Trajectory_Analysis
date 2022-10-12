from unpickle import unpickle
from ConfigSpace.ConfigSpace import ConfigSpace_AdditionalStates
from ConfigSpace.state_names import pre_final_state, color_dict, final_state, allowed_transition_attempts
from tqdm import tqdm
from Directories import df_sim_dir
from Directories import network_dir
import os
import json
from matplotlib import pyplot as plt
from itertools import groupby
import numpy as np
import pandas as pd

solver_geometry = {'ant': ('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'),
                   'pheidole': ('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'),
                   'human': ('MazeDimensions_human.xlsx', 'LoadDimensions_human.xlsx'),
                   'humanhand': ('MazeDimensions_humanhand.xlsx', 'LoadDimensions_humanhand.xlsx')}


time_step = 0.25  # seconds


class Path:
    """
    States is a class which represents the transitions of states of a trajectory. States are defined by eroding the CS,
    and then finding connected components.
    """

    def __init__(self, time_step: float, time_series=None, x=None, conf_space_labeled=None, only_states=False):
        """
        :param step: after how many frames I add a label to my label list
        :param x: trajectory
        :return: list of strings with labels
        """
        self.time_step = time_step
        self.time_series = time_series
        if self.time_series is None and x is not None:
            self.time_series = self.get_time_series(conf_space_labeled, x)
            # print('No correction')
            self.time_series = conf_space_labeled.correct_time_series(self.time_series, filename=x.filename)
            # self.save_transition_images(x)
            if only_states:
                self.time_series = [l[0] for l in self.time_series]
        # self.state_series = self.calculate_state_series(self.time_series, conf_space_labeled)
        self.state_series = None

    def get_time_series(self, conf_space_labeled, x):
        coords = [coords for coords in x.iterate_coords(time_step=self.time_step)]
        indices = [conf_space_labeled.coords_to_indices(*coords) for coords in coords]
        labels = [None]
        for i, index in enumerate(indices):
            labels.append(self.label_configuration(index, conf_space_labeled))
        labels = labels[1:]

        if pre_final_state in labels[-1] and pre_final_state != labels[-1]:
            labels.append(pre_final_state)

        return labels

    def label_configuration(self, index, conf_space_labeled) -> str:
        label = conf_space_labeled.space_labeled[index]
        if label == '0':
            label = conf_space_labeled.find_closest_state(index)
        # if set(label) == set('d'):
        #     conf_space_labeled.draw_ind(index)
        return label

    @staticmethod
    def combine_transitions(labels) -> list:
        """
        I want to combine states, that are [.... 'gb' 'bg'...] to [... 'gb'...]
        """
        labels = [''.join(sorted(state)) for state in labels]
        mask = [True] + [sorted(state1) != sorted(state2) for state1, state2 in zip(labels, labels[1:])]
        return np.array(labels)[mask].tolist()

    @staticmethod
    def symmetrize(state_series):
        return [state.replace('d', 'e') for state in state_series]

    @staticmethod
    def only_states(state_series):
        return [state[0] for state in state_series]

    def state_at_time(self, time: float) -> str:
        return self.time_series[int(time / self.time_step)]

    @staticmethod
    def calculate_state_series(time_series, conf_space_labeled):
        """
        Reduces time series to series of states. No self loops anymore.
        :return:
        """
        labels = [''.join(ii[0]) for ii in groupby([tuple(label) for label in time_series])]
        labels = Path.combine_transitions(labels)
        labels = conf_space_labeled.add_final_state(labels)
        return labels

    @staticmethod
    def time_stamped_series(time_series, time_step) -> list:
        groups = groupby(time_series)
        return [(label, sum(1 for _ in group) * time_step) for label, group in groups]

    @classmethod
    def create_dicts(cls, myDataFrame, ConfigSpace_class=ConfigSpace_AdditionalStates):
        dictio_ts = {}
        dictio_ss = {}
        shape = 'SPT'
        myDataFrame = myDataFrame[myDataFrame['shape'] == shape]

        for solver in myDataFrame['solver'].unique():
            print(solver)
            df = myDataFrame[myDataFrame['solver'] == solver].sort_values('size')
            groups = df.groupby(by=['size'])
            for size, cs_group in groups:
                cs_labeled = ConfigSpace_class(solver, size, shape, solver_geometry[solver])
                cs_labeled.load_labeled_space()
                for _, exp in tqdm(cs_group.iterrows()):
                    print(exp['filename'])
                    x = unpickle(exp['filename'])
                    path_x = Path(time_step=0.25, x=x, conf_space_labeled=cs_labeled)
                    dictio_ts[exp['filename']] = path_x.time_series
                    dictio_ss[exp['filename']] = path_x.state_series
        return dictio_ts, dictio_ss

    @staticmethod
    def find_missing(time_series_dict, myDataFrame, solver=None):
        myDataFrame = myDataFrame[myDataFrame['shape'] == 'SPT']

        if solver is not None:
            myDataFrame = myDataFrame[myDataFrame['solver'] == solver]

        to_add = myDataFrame[~myDataFrame['filename'].isin(time_series_dict.keys())]
        return to_add

    @classmethod
    def add_to_dict(cls, to_add, ConfigSpace_class, time_series_dict, state_series_dict, solver=None) -> tuple:
        """

        """
        dictio_ts = {}
        dictio_ss = {}

        solver_groups = to_add.groupby('solver')
        for solver, solver_group in solver_groups:
            size_groups = solver_group.groupby('size')
            for size, cs_group in size_groups:
                print(size)
                cs_labeled = ConfigSpace_class(solver, size, 'SPT', solver_geometry[solver])
                cs_labeled.load_labeled_space()
                for _, exp in tqdm(cs_group.iterrows()):
                    print(exp['filename'])
                    if (exp['maze dimensions'], exp['load dimensions']) != solver_geometry[solver]:
                        dictio_ts[exp['filename']] = None
                        dictio_ss[exp['filename']] = None
                    else:
                        x = unpickle(exp['filename'])
                        path_x = Path(time_step=0.25, x=x, conf_space_labeled=cs_labeled)
                        dictio_ts[exp['filename']] = path_x.time_series
                        dictio_ss[exp['filename']] = path_x.state_series
        time_series_dict.update(dictio_ts)
        state_series_dict.update(dictio_ss)
        return time_series_dict, state_series_dict

    def bar_chart(self, ax, axis_label='', winner=False, food=False, block=False):
        ts = self.time_series
        if 'i' in ts:
            ts.remove('i')

        dur = Path.time_stamped_series(ts, self.time_step)

        left = 0
        given_names = {}

        for name, duration in dur:
            dur_in_min = duration / 60
            if block:
                b = ax.barh(axis_label, 1, color=color_dict[name], left=left, label=name)
                left += 1
            else:
                b = ax.barh(axis_label, dur_in_min, color=color_dict[name], left=left, label=name)
                left += dur_in_min
            if name not in given_names:
                given_names.update({name: b})

        labels = list(color_dict.keys())
        handles = [plt.Rectangle((0, 0), 1, 1, color=color_dict[label]) for label in labels]
        plt.legend(handles, labels)

    @staticmethod
    def get_dicts(name=''):
        with open(os.path.join(network_dir, 'time_series' + name + '.json'), 'r') as json_file:
            time_series_dict = json.load(json_file)
            json_file.close()

        with open(os.path.join(network_dir, 'state_series' + name + '.json'), 'r') as json_file:
            state_series_dict = json.load(json_file)
            json_file.close()
        return time_series_dict, state_series_dict

    @staticmethod
    def save_dicts(time_series_dict, state_series_dict, name=''):
        with open(os.path.join(network_dir, 'time_series' + name + '.json'), 'w') as json_file:
            json.dump(time_series_dict, json_file)
            json_file.close()

        with open(os.path.join(network_dir, 'state_series' + name + '.json'), 'w') as json_file:
            json.dump(state_series_dict, json_file)
            json_file.close()


if __name__ == '__main__':
    df = pd.read_json(df_sim_dir)
    time_series_dict_selected_states, state_series_dict_selected_states = Path.create_dicts(df,
                                                                                            ConfigSpace_AdditionalStates)
    Path.save_dicts(time_series_dict_selected_states, state_series_dict_selected_states, name='_selected_states_sim')

    # time_dict, state_dict = Path.get_dicts(name='_selected_states_sim')

    # filename = 'M_SPT_4700022_MSpecialT_1_ants'
    # x = get(filename)
    # cs_labeled = ConfigSpace_AdditionalStates(x.solver, x.size, x.shape, x.geometry())
    # cs_labeled.load_labeled_space()
    # path = Path(time_step, x=x, conf_space_labeled=cs_labeled)
    # x.play(step=5, path=path, videowriter=True)

    # to_add = Path.find_missing(myDataFrame)
    # time_series_dict_selected_states, state_series_dict_selected_states = Path.add_to_dict(to_add,
    #                                                                                        ConfigSpace_class,
    #                                                                                        time_series_dict_selected_states,
    #                                                                                        state_series_dict_selected_states)
    # filenames = []
    # to_recalculate = myDataFrame[myDataFrame['filename'].isin(filenames)]
    # time_series_dict_selected_states, state_series_dict_selected_states = Path.add_to_dict(to_recalculate,
    #                                             ConfigSpace_class, time_series_dict_selected_states,
    #                                             state_series_dict_selected_states)

    # Path.save_dicts(time_series_dict_selected_states, state_series_dict_selected_states, name='_selected_states')
