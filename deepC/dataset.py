import os

import pandas as pd
import numpy as np

from torch.utils.data import Dataset

class VehicleDataset(Dataset): 
    def __init__(self, config, set='train'):
        self.partition = config['data']['partition'][set]
        self.data_dir = config['data']['data_dir']
        self.set = set
        self.n_frames = config['training']['n_frames']
    
    def __len__(self):
        return len(self.partition)
    
    def __getitem__(self, idx):
        filename, class_value = self.partition[idx]
        log_melspectrogram = np.load(os.path.join(self.data_dir + 'npy', self.set, filename.split('/')[-1].replace('.wav', '') + '.npy'))[:, :, :self.n_frames]
        return log_melspectrogram, class_value