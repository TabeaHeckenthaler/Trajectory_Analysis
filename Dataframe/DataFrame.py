import pandas as pd
from Directories import df_sim_dir, mini_SaverDirectories
from Dataframe.SingleExperiment import SingleExperiment
from tqdm import tqdm
import os


class DataFrame(pd.DataFrame):
    def __init__(self, input, columns=None):
        if type(input) is pd.DataFrame:
            super().__init__(input, columns=columns)

        elif type(input) is list:
            super().__init__(pd.concat(input).reset_index(), columns=columns)

    def __add__(self, df_2):
        return DataFrame(pd.concat([self, df_2], ignore_index=True))

    @classmethod
    def create(cls, solver_filenames):
        singleExperiments = []
        for solver, filenames in solver_filenames.items():
            for filename in tqdm(filenames):
                singleExperiments.append(SingleExperiment(filename, solver))
        df = pd.concat(singleExperiments).reset_index(drop=True)
        return df


if __name__ == '__main__':
    solver_filenames = {'pheidole': os.listdir(mini_SaverDirectories['pheidole'])}
    myDataFrame = DataFrame.create(solver_filenames)
    myDataFrame.to_json(df_sim_dir)
    print(myDataFrame.head())

    # myDataFrame_sim = pd.read_json(df_sim_dir)
