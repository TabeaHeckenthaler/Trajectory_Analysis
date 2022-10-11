from ConfigSpace.state_names import *
from Directories import PhaseSpaceDirectory
import numpy as np
import pickle
import os
from scipy import ndimage


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

    def load_labeled_space(self) -> None:
        """
        Load Phase Space pickle.
        Note for Tabea: in her code, these would be saved as
        M_SPT_MazeDimensions_new2021_SPT_ant_labeled_erosion_12_small for example, but here only M_SPT.pkl
        """
        directory = os.path.join(PhaseSpaceDirectory, self.shape, self.size + '_' + self.shape + '.pkl')

        if os.path.exists(directory):
            print('Loading labeled from ', directory, '.')
            self.space_labeled = pickle.load(open(directory, 'rb'))
            self.reduce_states()
            self.enlarge_transitions()
            self.split_states()

        else:
            raise ValueError('Cannot find directory ' + directory)

    def reduce_states(self):
        for name_initial, name_final in same_names:
            self.space_labeled[name_initial == self.space_labeled] = name_final

    @staticmethod
    def dilate(space: np.array, radius: int) -> np.array:
        """
        dilate phase space
        :param space:
        :param radius: radius of dilation
        """
        return np.array(ndimage.morphology.grey_dilation(space, size=tuple(radius for _ in range(space.ndim))),
                        dtype=bool)

    def enlarge_transitions(self):
        to_enlarge = {'be': 30, 'bf': 30, 'eb': 45, 'eg': 45, 'cg': 15}
        for state, radius in to_enlarge.items():
            mask = self.dilate(self.space_labeled == state, radius)
            mask = np.logical_and(mask, self.space_labeled == state[0])
            self.space_labeled[mask] = state

    def split_states(self):
        # split 'a'
        boundary = self.space_labeled.shape[2] / 4
        b_mask = np.zeros(self.space_labeled.shape, dtype=bool)
        b_mask[..., int(boundary):int(boundary*3)] = True
        a_mask = np.isin(self.space_labeled, ['a', 'ab', 'ac'])

        self.space_labeled[np.logical_and(a_mask, b_mask)] = 'ac'
        self.space_labeled[np.logical_and(a_mask, np.logical_not(b_mask))] = 'ab'

        # split 'bf'
        a_mask = self.space_labeled == 'bf'
        boundary = self.space_labeled.shape[1] // 2 + 1
        b_mask = np.zeros(self.space_labeled.shape, dtype=bool)
        b_mask[:, :int(boundary)] = True

        self.space_labeled[np.logical_and(a_mask, b_mask)] = 'b1'
        self.space_labeled[np.logical_and(a_mask, np.logical_not(b_mask))] = 'b2'

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


if __name__ == '__main__':
    DEBUG = 1
    solver, size, shape, geometry = ('ant', 'M', 'SPT',
                                     ('MazeDimensions_new2021_SPT_ant.xlsx', 'LoadDimensions_new2021_SPT_ant.xlsx'))

    cs_labeled = ConfigSpace_AdditionalStates(solver, size, shape, geometry)
    cs_labeled.load_labeled_space()
