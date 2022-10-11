from os import path
import os

# directory where your simulated trajectories are saved
with open('data_directory.txt') as f:
    lines = f.readlines()
data_home = lines[0]

PhaseSpaceDirectory = path.join(data_home, 'Configuration_Spaces')
work_dir = path.join(data_home, 'Pickled_Trajectories')
SaverDirectories = {'ant': {True: path.join(work_dir, 'Ant_Trajectories', 'Free'),
                            False: path.join(work_dir, 'Ant_Trajectories', 'Slitted')},
                    'pheidole': path.join(work_dir, 'Pheidole_Trajectories'),
                    'human': path.join(work_dir, 'Human_Trajectories'),
                    'humanhand': path.join(work_dir, 'HumanHand_Trajectories'),
                    'gillespie': path.join(work_dir, 'Gillespie_Trajectories'),
                    'ps_simulation': path.join(work_dir, 'PS_simulation_Trajectories')}

# directory where the results from the states analysis will be saved
network_dir = path.join(os.getcwd(), 'States')

# directory where your DataFrame is saved, which contains all names of your trajectories
df_sim_dir = path.join(data_home, 'DataFrame', 'data_frame_sim.json')
