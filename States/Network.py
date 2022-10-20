import pathpy
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from copy import copy

state_order = ['b2', 'b1', 'be', 'b', 'ab', 'ac', 'cg', 'c', 'eg', 'eb', 'e', 'f', 'h']


class Network(pathpy.Network):
    @classmethod
    def init_from_paths(cls, paths, solver, shape, size=None):
        self = super(Network, cls).from_paths(paths)

        # if size is not None and 'Small' in size:
        #     size = 'Small'

        if size is None:
            size_string = ''
        else:
            size_string = size

        self.name = '_'.join(['network', solver, size_string, shape])
        self.paths = paths
        self.add_edges(paths)
        return self

    def add_edges(self, paths: pathpy.Paths):
        for states, weight in paths.paths[1].items():
            if len(states) == 2:
                self.add_edge(states[0], states[1], weight=weight[0])

    @staticmethod
    def swap(m: pd.DataFrame, row2: int, row1: int):
        order = m.index.tolist()
        order[row2], order[row1] = copy(order[row1]), copy(order[row2])
        m = m.reindex(columns=order, index=order)
        return m

    def find_P(self, T):
        """
        reorganize matrix to canonical form
        """
        num_absorbing = 0
        for i, r in enumerate(T.index):
            if 1.0 == T.loc[r][r]:
                num_absorbing += 1
                for ii in range(num_absorbing, i + 1)[::-1]:
                    T = self.swap(T, ii, ii - 1)
        return T, num_absorbing

    def markovian_analysis(self) -> tuple:
        T = pd.DataFrame(self.transition_matrix().toarray().transpose(),
                         columns=list(self.node_to_name_map()),
                         index=list(self.node_to_name_map()))
        final_state = 'i'
        if final_state in T.columns:
            T[final_state][final_state] = 1

        P, num_absorbing = self.find_P(T)
        Q, R = P.iloc[num_absorbing:, num_absorbing:], P.iloc[num_absorbing:, 0:num_absorbing]
        transient_state_order = P.columns[num_absorbing:]

        t = None
        if num_absorbing > 0:
            N = pd.DataFrame(np.linalg.inv(np.identity(Q.shape[-1]) - Q),
                             columns=transient_state_order,
                             index=transient_state_order
                             )  # fundamental matrix
            t = np.matmul(N, np.ones(N.shape[0]))
            B = pd.DataFrame(np.matmul(N.to_numpy(), R.to_numpy()),
                             index=transient_state_order,
                             columns=T.index[-num_absorbing:]
                             )  # absorption probabilities
        return T, t

    @classmethod
    def absorption_time(cls, paths):
        states = {state: [] for state in state_order}

        absorbed_paths, non_absorbed_paths = [], []

        for p in paths:
            if p[-1] != 'i':
                non_absorbed_paths.append(p)
            else:
                absorbed_paths.append(p)

        for p in absorbed_paths:
            for state in state_order:
                state_locations = np.where(np.array(p) == state)[0]
                time_until_end = len(p) * np.ones_like(state_locations) - state_locations
                states[state] = states[state] + list(time_until_end)

        for p in non_absorbed_paths:
            for state in state_order:
                state_locations = np.where(np.array(p) == state)[0]
                time_until_end = (len(p) + np.mean(states[p[-1]], dtype=int)) * np.ones_like(state_locations) - state_locations
                states[state] = states[state] + list(time_until_end)

        absorption_times = {state: (np.mean(list_of_times), np.std(list_of_times) / np.sqrt(np.size(list_of_times)))
                                  for state, list_of_times in states.items()}

        return absorption_times
