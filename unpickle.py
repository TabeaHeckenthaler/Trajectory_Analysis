from Directories import mini_work_dir
import os
from os import path
import pickle
from trajectory import Trajectory


def unpickle(filename):
    """
    Allows the loading of saved trajectory objects.
    :param filename: Name of the trajectory that is supposed to be un-pickled
    :return: trajectory object
    """
    for root, dirs, files in os.walk(mini_work_dir):
        for dir in dirs:
            if filename in os.listdir(path.join(root, dir)):
                address = path.join(root, dir, filename)
                with open(address, 'rb') as f:
                    shape, size, solver, filename, fps, position, angle, frames, winner = pickle.load(f)
                return Trajectory(shape=shape, size=size, solver=solver, filename=filename, position=position,
                                  angle=angle, frames=frames, winner=winner)
    else:
        raise ValueError('I cannot find ' + filename)


if __name__ == '__main__':
    x = unpickle('M_SPT_4700014_MSpecialT_1_ants (part 1)')
