from os import path

project_home = path.join(path.abspath(__file__).split('\\')[0] + path.sep, *path.abspath(__file__).split(path.sep)[1:-1])

# directory where your simulated trajectories are saved
with open(path.join(project_home, 'data_directory.txt')) as f:
    lines = f.readlines()
data_home = lines[0]

# PhaseSpaceDirectory = path.join(data_home, 'Configuration_Spaces')
PhaseSpaceDirectory = path.join(project_home, 'ConfigSpace', 'CSs')

work_dir = path.join(data_home, 'Pickled_Trajectories')
# SaverDirectories = {'ant': {True: path.join(work_dir, 'Ant_Trajectories', 'Free'),
#                             False: path.join(work_dir, 'Ant_Trajectories', 'Slitted')},
#                     'pheidole': path.join(work_dir, 'Pheidole_Trajectories'),
#                     'human': path.join(work_dir, 'Human_Trajectories'),
#                     'humanhand': path.join(work_dir, 'HumanHand_Trajectories'),
#                     'gillespie': path.join(work_dir, 'Gillespie_Trajectories'),
#                     'ps_simulation': path.join(work_dir, 'PS_simulation_Trajectories')}

# directory where the results from the states analysis will be saved
network_dir = path.join(project_home, 'States')

# directory where your Dataframe is saved, which contains all names of your trajectories
df_sim_dir = path.join(data_home, 'Dataframe', 'data_frame_test_sim.json')

mini_work_dir = path.join(data_home, 'mini_Pickled_Trajectories')
mini_SaverDirectories = {'ant': path.join(mini_work_dir, 'Ant_Trajectories'),
                         'pheidole': path.join(mini_work_dir, 'Pheidole_Trajectories'),
                         'human': path.join(mini_work_dir, 'Human_Trajectories'),
                         'humanhand': path.join(mini_work_dir, 'HumanHand_Trajectories'),
                         'gillespie': path.join(mini_work_dir, 'Gillespie_Trajectories'),
                         'ps_simulation': path.join(mini_work_dir, 'PS_simulation_Trajectories')}