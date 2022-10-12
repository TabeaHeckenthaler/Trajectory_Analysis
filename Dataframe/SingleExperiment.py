import pandas as pd
from unpickle import unpickle

length_unit = {'ant': 'cm', 'human': 'm', 'humanhand': 'cm', 'ps_simulation': 'cm', 'gillespie': 'cm', 'pheidole': 'cm'}


def length_unit_func(solver):
    return length_unit[solver]


class SingleExperiment(pd.DataFrame):
    def __init__(self, filename, solver, df: pd.DataFrame = None):
        if df is None:
            super().__init__([[filename, solver]], columns=['filename', 'solver'])
            self.add_information()
        else:
            super().__init__(df)

    def add_information(self):
        x = unpickle(self['filename'][0])
        self['size'] = str(x.size)
        self['solver'] = x.solver
        self['shape'] = str(x.shape)
        self['winner'] = bool(x.winner)
        self['fps'] = int(x.fps)
        # self['communication'] = bool(x.communication)
        # self['length unit'] = str(length_unit_func(x.solver))
        # self['initial condition'] = str(x.initial_cond())
        # self['force meter'] = bool(x.has_forcemeter())
        # self['maze dimensions'], self['load dimensions'] = x.geometry()
        # self['counted carrier number'] = None
        # self['time [s]'] = x.timer()
        # self['comment'] = ''
