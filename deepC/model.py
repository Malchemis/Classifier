import os 
import argparse
import yaml
import pickle

from dataset import VehicleDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary


class ConvBlock(nn.Module):
    def __init__(self, input_features, output_features, kernel, padding, stride):
        super(ConvBlock,self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.kernel = kernel
        self.padding = padding
        self.stride = stride
        
        self.conv = nn.Conv2d(in_channels=input_features, out_channels=output_features, kernel_size=kernel, padding=padding, stride=stride)
        self.bNorm= nn.BatchNorm2d(output_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.conv(x)
        output = self.bNorm(output)
        output = self.relu(output)
        return output


class VGG(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(VGG, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        layers = []
        fc_layers = []
        base_features = 32

        layers.append(ConvBlock(input_features=num_channels, output_features=base_features, kernel=3, padding=1, stride=1))
        layers.append(ConvBlock(input_features=base_features, output_features=2*base_features, kernel=3, padding=1, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        layers.append(ConvBlock(input_features=2*base_features, output_features=2*base_features, kernel=3, padding=1, stride=1))
        layers.append(ConvBlock(input_features=2*base_features, output_features=4*base_features, kernel=3, padding=1, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        layers.append(ConvBlock(input_features=4*base_features, output_features=4*base_features, kernel=3, padding=1, stride=1))
        layers.append(ConvBlock(input_features=4*base_features, output_features=8*base_features, kernel=3, padding=1, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        layers.append(ConvBlock(input_features=8*base_features, output_features=8*base_features, kernel=3, padding=1, stride=1))
        layers.append(ConvBlock(input_features=8*base_features, output_features=8*base_features, kernel=3, padding=1, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        layers.append(nn.AdaptiveAvgPool2d(2)) # Tester 7 comme dans VGG 

        # fc_layers.extend([nn.Linear(in_features=2*2*(8*base_features), out_features= base_features*base_features), nn.ReLU(), nn.Dropout(0.5)])
        fc_layers.extend([nn.Linear(in_features=base_features, out_features= base_features*base_features), nn.ReLU(), nn.Dropout(0.5)])
        fc_layers.extend([nn.Linear(in_features=base_features*base_features, out_features= base_features*base_features), nn.ReLU(), nn.Dropout(0.5)])
        fc_layers.extend([nn.Linear(in_features=base_features*base_features, out_features= self.num_classes), nn.Softmax(dim=1)])

        self.layers = nn.Sequential(*layers)
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        output = self.layers(x)
        output = output.view(output.size(0), -1)
        output = self.fc_layers(output)
        return output
    
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

    num_channels = 1
    num_classes = len(config['data']['classes'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgg = VGG(num_channels, num_classes).to(device)
    print(vgg)
    
    train_dataloader = DataLoader(VehicleDataset(partition, config), batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'])
    print(f"Shape of the first batch: {next(iter(train_dataloader))[0].shape}")
    summary(vgg, train_dataloader.dataset[0][0].shape)