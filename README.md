How to currently use this project

1) Create a file data_directory.txt in the project_folder, which contains the name of the directory 
in which your trajectories are saved. On Tabea's computer the .txt file contains 
\\phys-guru-cs\ants\Tabea\PyCharm_Data\AntsShapes. 
In this directory, there should be a folder 'mini_Pickled_Trajectories/Gillespie_Trajectories'. 
This folder should contain .pkl files which contain pickles of the trajectories and can be unpickled 
by the module unpickle.py. 

2) Add the XL_SPT.pkl, L_SPT.pkl, M_SPT.pkl and S_SPT.pkl files to the ConfigSpace/CSs/SPT folder. 

3) Run DataFrame.py found in ./Dataframe. 
This should save a pd.DatFrame in df_sim_dir (defined in Directories.py).
This Dataframe contains the names of the trajectories that should be analyzed.  

4) Run Path.py in order to save the state names for all trajectories.  
The are saved in trajectory_states_dir (defined in Directories.py).

5) Run main.py. This script will import all the necessary file and plot the states for every experiment. 

6) Lets try to understand what further analysis we can do :)  