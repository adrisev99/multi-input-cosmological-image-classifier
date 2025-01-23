import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# name of the file
fmaps = r'.\Maps_Mcdm_IllustrisTNG_LH_z=0.00.npy'
fparams = r'.\params_LH_IllustrisTNG.txt'

# read the data
maps = np.load(fmaps)
params  = np.loadtxt(fparams)


selected_indices = np.random.choice(len(maps), 5, replace=False)

output_folder = r'C:\Users\adria\Documents\VSCode Projects\JHU24\CMDs\MapImages'
os.makedirs(output_folder, exist_ok=True)


columns = ['Map_ID', 'Omega_m', 'sigma_8', 'A_SN1', 'A_AGN1', 'A_SN2', 'A_AGN2']
params_df = pd.DataFrame(columns=columns)

for map_number in selected_indices:
    params_map = params[map_number // 15]
    plt.imshow(np.log10(maps[map_number]), cmap=plt.get_cmap('binary_r'), origin='lower', interpolation='bicubic')

    plt.axis('off')
    

    output_path = os.path.join(output_folder, f'Map_{map_number}.png')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    params_df.loc[map_number] = [map_number] + list(params_map)

csv_path = os.path.join(output_folder, 'map_parameters.csv')
params_df.to_csv(csv_path, index = False)
