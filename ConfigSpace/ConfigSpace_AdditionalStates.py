from Directories import PhaseSpaceDirectory, project_home
import numpy as np
import pickle
from ConfigSpace.state_names import *
import pandas as pd
import ast
from skfmm import distance
from itertools import groupby
from os import path

measurements = pd.read_excel(project_home + '\\ConfigSpace\\CSs\\SPT\\CS_measurements.xlsx')


class ConfigSpace_AdditionalStates:
    """
    This class stores configuration space for a piano_movers problem in a 3 dim array.
    Axis 0 = x direction
    Axis 1 = y direction
    Axis 0 = x direction
    Every element of self.space carries on of the following indices_to_coords of:
    - '0' (not allowed)
    - A, B...  (in self.eroded_space), where N is the number of states
    - n_1 + n_2 where n_1 and n_2 are in (A, B...).
        n_1 describes the state you came from.
        n_2 describes the state you are
    """

    def __init__(self, solver, size, shape, geometry):
        self.solver = solver
        self.shape = shape
        self.size = size
        self.geometry = geometry
        self.space_labeled = None
        m = measurements[measurements['size'] == self.size].iloc[0]
        self.pos_resolution = m['pos_resolution']
        self.theta_resolution = m['theta_resolution']
        self.extent = ast.literal_eval(m['extent'])

    def load_labeled_space(self) -> None:
        """
        Load Phase Space pickle.
        Note for Tabea: in her code, these would be saved as
        M_SPT_MazeDimensions_new2021_SPT_ant_labeled_erosion_12_small for example, but here only M_SPT.pkl
        """

        directory = path.join(PhaseSpaceDirectory, self.shape, self.size + '_' + self.shape + '.pkl')
        print('Loading labeled from ', directory, '.')
        self.space_labeled = pickle.load(open(directory, 'rb'))

    def find_closest_state(self, index: list) -> str:
        """
        :return: name of the ps_state closest to indices_to_coords, chosen from ps_states
        """
        index_theta = index[2]
        found_close_state = False
        border = 5
        while not found_close_state:
            if border > 100:
                raise ValueError('Cant find closest state: ' + str(index))
            if index_theta - border < 0 or index_theta + border > self.space_labeled.shape[2]:
                cut_out = np.concatenate([self.space_labeled[
                                          max(0, index[0] - border):index[0] + border,
                                          max(0, index[1] - border):index[1] + border,
                                          (index_theta - border) % self.space_labeled.shape[2]:],

                                          self.space_labeled[max(0, index[0] - border):index[0] + border,
                                          max(0, index[1] - border):index[1] + border,
                                          0:(index_theta + border) % self.space_labeled.shape[2]]
                                          ], axis=-1)
            else:
                cut_out = self.space_labeled[max(0, index[0] - border):index[0] + border,
                          max(0, index[1] - border):index[1] + border,
                          index[2] - border:index[2] + border]
            states = np.unique(cut_out).tolist()

            if '0' in states:
                states.remove('0')

            if len(states) > 0:
                found_close_state = True
            else:
                border += 10
        if len(states) == 1:
            return states[0]

        distances = {}
        values, counts = np.unique(cut_out, return_counts=True)
        d = {key: value for key, value in zip(values, counts)}
        d.pop('0')
        spotty = np.array(list(d.values()) / sum(list(d.values()))) < 0.1
        to_pop = [key for s, key in zip(spotty, d.keys()) if s]
        d = {key: d[key] for key in d.copy().keys() if key not in to_pop}

        for state in d.keys():
            # if np.sum(np.array(list(d.values())) > 20) == 1:
            #     return list(d.keys())[np.where(np.array(list(d.values())) > 20)[0][0]]
            distances[state] = self.calculate_distance(cut_out == state, np.ones(shape=cut_out.shape, dtype=bool))[border, border, border]
        return min(distances, key=distances.get)

    @staticmethod
    def calculate_distance(zero_distance_space: np.array, available_states: np.array) -> np.array:
        """
        Calculate the distance of every node in mask to space.
        :param zero_distance_space: Area, which has zero distance.
        :param available_states: Nodes, where distance to the zero_distance_space should be calculated.
        :return: np.array with distances from each node
        """

        # self.distance = distance(np.array((~np.array(self.space, dtype=bool)), dtype=int), periodic=(0, 0, 1))
        phi = np.array(~zero_distance_space, dtype=int)
        masked_phi = np.ma.MaskedArray(phi, mask=~available_states)
        d = distance(masked_phi, periodic=(0, 0, 1))
        return d

    @staticmethod
    def valid_state_transition(s1, s2) -> bool:
        for c in connected:
            if s1 in c and s2 in c:
                return True
        return False

    @classmethod
    def add_missing_transitions(cls, labels) -> list:
        """
        I want to correct states series, that are [.... 'g' 'b'...] to [... 'g' 'gb' 'b'...]
        """
        new_labels = [labels[0]]

        for ii, state2 in enumerate(labels[1:]):
            state1 = new_labels[-1]
            if not cls.valid_state_transition(state1, state2):
                if state1 in ['ac'] and 'e' in state2:
                    new_labels.append('c')  # only for small SPT ants
            else:
                new_labels.append(state2)
        return new_labels

    def coords_to_index(self, axis: int, value):
        """
        Translating coords to index of axis
        :param axis: What axis is coordinate describing
        :param value:
        :return:
        """
        if value is None:
            return None
        resolut = {0: self.pos_resolution, 1: self.pos_resolution, 2: self.theta_resolution}[axis]
        value_i = min(int(np.round((value - list(self.extent.values())[axis][0]) / resolut)),
                      self.space_labeled.shape[axis] - 1)

        if value_i >= self.space_labeled.shape[axis] or value_i <= -1:
            print('check', list(self.extent.keys())[axis])
        return value_i

    def coords_to_indices(self, x: float, y: float, theta: float) -> tuple:
        """
        convert coordinates into indices_to_coords in PhaseSpace
        :param x: x position of CM in cm
        :param y: y position of CM in cm
        :param theta: orientation of axis in radian
        :return: (xi, yi, thetai)
        """
        return self.coords_to_index(0, x), self.coords_to_index(1, y), \
               self.coords_to_index(2, theta % (2 * np.pi))

    @classmethod
    def correct_time_series(cls, time_series, filename=''):
        time_series = cls.add_initial_state(time_series)
        time_series = cls.add_final_state(time_series)
        time_series = cls.add_missing_transitions(time_series)
        time_series = cls.delete_false_transitions(time_series, filename=filename)
        time_series = cls.get_rid_of_short_lived_states(time_series)
        return time_series

    @staticmethod
    def add_final_state(labels):
        # I have to open a new final state called i, which is in PS equivalent to h, but is an absorbing state, meaning,
        # it is never left.
        times_in_final_state = np.where(np.array(labels) == pre_final_state)[0]
        if len(times_in_final_state) > 0:
            # print(labels[:first_time_in_final_state[0] + 1][-10:])
            return labels + [final_state]
        else:
            return labels

    @classmethod
    def get_rid_of_short_lived_states(cls, labels, min=5):
        grouped = [(''.join(k), sum(1 for _ in g)) for k, g in groupby([tuple(label) for label in labels])]
        new_labels = [grouped[0][0] for _ in range(grouped[0][1])]
        for i, (label, length) in enumerate(grouped[1:-1], 1):
            if length <= min and cls.valid_state_transition(new_labels[-1], grouped[i + 1][0]):
                # print(grouped[i - 1][0] + ' => ' + grouped[i + 1][0])
                new_labels = new_labels + [new_labels[-1] for _ in range(length)]
            else:
                new_labels = new_labels + [label for _ in range(length)]
        new_labels = new_labels + [grouped[-1][0] for _ in range(grouped[-1][1])]
        return new_labels

    @staticmethod
    def add_initial_state(labels):
        # if the tracking only started, once the shape was already in b
        if labels[0] == 'b':
            return ['ab'] + labels
        else:
            return labels

    @staticmethod
    def necessary_transitions(state1, state2, ii: int = '') -> list:
        if state1 == 'c' and state2 == 'fh':
            return ['ce', 'e', 'ef', 'f']
        if state1 == 'c' and state2 == 'f':
            return ['ce', 'e', 'ef']
        if state1 == 'ba' and state2 == 'cg':
            return ['a', 'ac', 'c']

        # otherwise, our Markov chain is not absorbing for L ants
        if set(state1) in [set('ef'), set('ec')] and set(state1) in [set('ef'), set('ec')]:
            return ['e']

        if len(state1) == len(state2) == 1:
            transition = ''.join(sorted(state1 + state2))
            if transition in allowed_transition_attempts:
                return [transition]
            else:
                print('Skipped 3 states: ' + state1 + ' -> ' + state2 + ' in ii ' + str(ii))
                return []

        elif len(state1) == len(state2) == 2:
            print('Moved from transition to transition: ' + state1 + '_' + state2 + ' in ii ' + str(ii))
            return []

        elif ''.join(sorted(state1 + state2[0])) in allowed_transition_attempts:
            return [''.join(sorted(state1 + state2[0])), state2[0]]
        elif ''.join(sorted(state1[0] + state2)) in allowed_transition_attempts:
            return [state1[0], ''.join(sorted(state1[0] + state2))]

        elif len(state2) > 1 and ''.join(sorted(state1 + state2[1])) in allowed_transition_attempts:
            return [''.join(sorted(state1 + state2[1])), state2[1]]
        elif len(state1) > 1 and ''.join(sorted(state1[1] + state2)) in allowed_transition_attempts:
            return [state1[1], ''.join(sorted(state1[1] + state2))]
        else:
            print('What happened: ' + state1 + ' -> ' + state2 + ' in ii ' + str(ii))
            return []

    @classmethod
    def delete_false_transitions(cls, labels, filename=''):
        old_labels = labels.copy()
        end_state = old_labels[-1]
        new_labels = [labels[0]]
        error_count = []
        for ii, next_state in enumerate(labels[1:], start=1):
            if not cls.valid_state_transition(new_labels[-1], next_state):
                error_count.append([new_labels[-1], next_state])
                new_labels.append(new_labels[-1])
            else:
                new_labels.append(next_state)
        if end_state != labels[-1]:
            print('Warning: end is not the same')
        return new_labels


if __name__ == '__main__':
    DEBUG = 1
    solver, size, shape, geometry = ('ant', 'M', 'SPT',
                                     ('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'))

    cs_labeled = ConfigSpace_AdditionalStates(solver, size, shape, geometry)
    cs_labeled.load_labeled_space()
