from os import path
import os

# data_home = path.join(path.sep + path.sep + 'phys-guru-cs', 'ants', 'Tabea', 'PyCharm_Data', 'AntsShapes')
with open('data_directory.txt') as f:
    lines = f.readlines()
data_home = lines[0]
home = os.getcwd()

PhaseSpaceDirectory = path.join(data_home, 'Configuration_Spaces')

work_dir = path.join(data_home, 'Pickled_Trajectories')


SaverDirectories = {'ant': {True: path.join(work_dir, 'Ant_Trajectories', 'Free'),
                            False: path.join(work_dir, 'Ant_Trajectories', 'Slitted')},
                    'pheidole': path.join(work_dir, 'Pheidole_Trajectories'),
                    'human': path.join(work_dir, 'Human_Trajectories'),
                    'humanhand': path.join(work_dir, 'HumanHand_Trajectories'),
                    'gillespie': path.join(work_dir, 'Gillespie_Trajectories'),
                    'ps_simulation': path.join(work_dir, 'PS_simulation_Trajectories')}

network_dir = path.join(home, 'States')

df_sim_dir = path.join(data_home, 'DataFrame', 'data_frame_sim.json')
