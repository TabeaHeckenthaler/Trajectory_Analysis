import numpy as np


class Trajectory:
    def __init__(self, size=None, shape=None, solver=None, filename=None, fps=50, winner=bool, position=None,
                 angle=None, frames=None):
        self.shape = shape  # shape (maybe this will become name of the maze...) (H, I, T, SPT)
        self.size = size  # size (XL, SL, L, M, S, XS)
        self.solver = solver  # ant, human, sim, humanhand
        self.filename = filename  # filename: shape, size, path length, sim/ants, counter
        self.fps = fps  # frames per second
        self.position = position
        self.angle = angle
        self.frames = frames
        self.winner = winner  # whether the shape crossed the exit

    def iterate_coords(self, time_step: float = 1) -> iter:
        """
        Iterator over (x, y, theta) of the trajectory, time_step is given in seconds
        :return: tuple (x, y, theta) of the trajectory
        """
        number_of_frames = self.angle.shape[0]
        length_of_movie_in_seconds = number_of_frames/self.fps
        len_of_slicer = np.floor(length_of_movie_in_seconds/time_step).astype(int)

        slicer = np.cumsum([time_step*self.fps for _ in range(len_of_slicer)][:-1]).astype(int)
        for pos, angle in zip(self.position[slicer], self.angle[slicer]):
            yield pos[0], pos[1], angle