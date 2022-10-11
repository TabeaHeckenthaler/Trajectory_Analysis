from Directories import work_dir
import os
from os import path
import pickle


def unpickle(filename):
    """
    Allows the loading of saved trajectory objects.
    :param filename: Name of the trajectory that is supposed to be un-pickled
    :return: trajectory object
    """
    for root, dirs, files in os.walk(work_dir):
        for dir in dirs:
            if filename in os.listdir(path.join(root, dir)):
                address = path.join(root, dir, filename)
                with open(address, 'rb') as f:
                    x = pickle.load(f)
                return x
    else:
        raise ValueError('I cannot find ' + filename)
