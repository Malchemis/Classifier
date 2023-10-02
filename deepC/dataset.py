import os
import yaml
import argparse
import pickle

import numpy as np

from torch.utils.data import Dataset, DataLoader

class VehicleDataset(Dataset): 
    def __init__(self, partition, config, set='train'):
        self.partition = partition[set]
        self.data_dir = config['data']['data_dir']
        self.set = set
        self.n_frames = config['training']['n_frames']
    
    def __len__(self):
        return len(self.partition)
    
    def __getitem__(self, idx):
        filename, class_value = self.partition[idx]
        log_melspectrogram = np.load(os.path.join(self.data_dir, 'npy', self.set, filename.split('/')[-1].replace('.wav', '') + '.npy'))[:, :, :self.n_frames]
        return log_melspectrogram, class_value
    
if __name__ == "__main__": 

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='conf/conf.yaml', help='Path to config file.')
    args = parser.parse_args()

    # Open the config file 
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(f'Loaded config from {args.config}')

    if os.path.exists(config['data']['partition']):
        with open(config['data']['partition'], 'rb') as f:
            partition = pickle.load(f)
        print(f'Loaded partitions from {config["data"]["partition"]}')
    else:
        raise ValueError(f'No partitions found at {config["data"]["partition"]}')
    
    train_dataloader = DataLoader(VehicleDataset(partition, config), batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'])

    print(f"Shape of the first batch: {next(iter(train_dataloader))[0].shape}")