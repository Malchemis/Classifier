import argparse
import os
import yaml
import pickle
from tqdm import tqdm

from statistics import mean
import matplotlib.pyplot as plt

from dataset import VehicleDataset
from model import VGG

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchsummary import summary


def train(model, optimizer, loader, writer, save_path= 'best_model.pt', epochs=10):
    criterion = torch.nn.CrossEntropyLoss()
    corrects = 0
    total = 0
    best_val_acc = 0
    train_acc_list = []
    val_acc_list = []
    for epoch in range(epochs):
        print(f'Epoch {epoch}/{epochs}')
        running_loss = []
        data = tqdm(loader)
        for features, labels in data:
            features, labels = features.to(device), labels.to(device)
            # Forward pass
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            # Statistics to compute accuracy and loss
            corrects += preds.eq(labels).sum().item()
            total += labels.size(0)
            loss = criterion(outputs, labels)
            running_loss.append(loss.item())
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            data.set_description(f'training loss: {mean(running_loss)}')
        accuracy = corrects / total
        train_acc_list.append(accuracy)
        writer.add_scalar('training loss', mean(running_loss), epochs)
        writer.add_scalar('training accuracy', accuracy, epochs)
        print(f'Training accuracy:{accuracy}')

        # Validation step 
        val_acc = test(model, val_dataloader)
        val_acc_list.append(val_acc)
        print(f'Validation accuracy:{val_acc}')
        writer.add_scalar('validation accuracy', val_acc, epochs)
        writer.flush()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_path))

        return train_acc, val_acc

def test(model, dataloader):
    test_corrects = 0
    total = 0
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            preds = model(features).argmax(1)
            test_corrects += preds.eq(labels).sum().item()
            total += labels.size(0)
    return test_corrects / total

if __name__ == "__main__": 

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='conf/conf.yaml', help='Path to config file.')
    parser.add_argument('--only_test', default=False, help='Only test the model.')
    args = parser.parse_args()

    # Open the config file 
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(f'Loaded config from {args.config}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    if os.path.exists(config['data']['partition']):
        with open(config['data']['partition'], 'rb') as f:
            partition = pickle.load(f)
        print(f'Loaded partitions from {config["data"]["partition"]}')
    else:
        raise ValueError(f'No partitions found at {config["data"]["partition"]}')
    
    # Create dataloaders
    train_dataloader = DataLoader(VehicleDataset(partition, config), batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'])
    val_dataloader = DataLoader(VehicleDataset(partition, config, set='val'), batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'])
    test_dataloader = DataLoader(VehicleDataset(partition, config, set='test'), batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'])
    
    # Create model
    num_channels = 1
    num_classes = len(config['data']['classes'])
    model = VGG(num_channels, num_classes).to(device)
    print(model)
    summary(model, train_dataloader.dataset[0][0].shape)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
    writer = SummaryWriter()

    if not os.path.exists('weights'):
        os.makedirs('weights')
    path_weights = os.path.join('weights', config['data']['dataset'] + '_best_model.pt')

    # Train the model
    if not args.only_test:

        train_acc, val_acc = train(model, optimizer, train_dataloader, writer, save_path=path_weights, epochs=config['training']['epochs'])

        # Plot the training and validation accuracy and save it 
        plt.plot(train_acc, color='b', label='Training accuracy')
        plt.plot(val_acc, color='r', label='Validation accuracy')
        plt.legend()
        plt.show()
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.savefig(os.path.join('plots', config['data']['dataset'] + '_accuracy.png'))
        plt.close()

        # Load the best model and test it
        model.load_state_dict(torch.load(path_weights))
        test_acc = test(model, test_dataloader)
        print(f'Test accuracy:{test_acc}')

    else:
        # Load the best model and test it
        model.load_state_dict(torch.load(path_weights))
        test_acc = test(model, test_dataloader)
        print(f'Test accuracy:{test_acc}')