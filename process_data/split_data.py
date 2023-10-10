import numpy as np
from glob import glob

files = glob('./*npy')
files = sorted(files, key=lambda x:int(x.split('_')[-1].split('.')[0]))

np_data_all = np.array([])
for file in files:
    temp = np.load(file, allow_pickle=True)
    np_data_all = np.append(np_data_all, temp)
    print(len(np_data_all))
    
    
for i in range(0, len(np_data_all), 200):
    print(f'saving npy_data_{i//200}.npy...')
    start = i
    end = min(i+200, len(np_data_all))
    np.save(f'npy_data/npy_data_{i//200}.npy', np_data_all[start:end])