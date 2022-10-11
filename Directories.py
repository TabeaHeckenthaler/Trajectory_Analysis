from os import path

data_home = path.join(path.sep + path.sep + 'phys-guru-cs', 'ants', 'Tabea', 'PyCharm_Data', 'AntsShapes')
PhaseSpaceDirectory = path.join(data_home, 'Configuration_Spaces')

home = path.join(path.abspath(__file__).split('\\')[0] + path.sep, *path.abspath(__file__).split(path.sep)[1:-1])

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
