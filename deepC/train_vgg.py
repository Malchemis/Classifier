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
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchsummary import summary


def train(config, model, optimizer, train_dataloader, val_dataloader, writer, save_path= 'best_model.pt', epochs=10):
    criterion = torch.nn.CrossEntropyLoss()
    best_val_acc = 0
    train_acc_list = []
    val_acc_list = []
    train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=len(config['data']['classes']), average='macro').to(device)
    val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=len(config['data']['classes']), average='macro').to(device)
    for epoch in range(epochs):
        print(f'Epoch {epoch}/{epochs} :')
        running_loss = []
        train_data = tqdm(train_dataloader)
        for features, labels in train_data:
            features, labels = features.to(device), labels.to(device)
            # Forward pass
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            # Statistics to compute accuracy and loss
            batch_train_acc = train_acc(preds, labels)
            loss = criterion(outputs, labels)
            running_loss.append(loss.item())
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_data.set_description(f'training loss: {mean(running_loss)}')
        # Training accuracy of the epoch
        total_train_acc = train_acc.compute()
        writer.add_scalar('training loss', mean(running_loss), epochs)
        print(f'Training accuracy:{total_train_acc}')
        train_acc_list.append(total_train_acc)


        # Validation step 
        for features, labels in val_dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            batch_val_acc = val_acc(preds, labels)
        
        total_val_acc = val_acc.compute()
        val_acc_list.append(total_val_acc)
        print(f'Validation accuracy:{total_val_acc}')

        if total_val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_path))

    return train_acc_list, val_acc_list

def test(config, model, test_dataloader):
    test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=len(config['data']['classes']), average='macro').to(device)
    f1_score = torchmetrics.F1Score(task='multiclass', num_classes=len(config['data']['classes']), average='macro').to(device)
    with torch.no_grad():
        for features, labels in test_dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            batch_test_acc = test_acc(preds, labels)
            batch_f1_score = f1_score(preds, labels)
    return test_acc.compute(), f1_score.compute()

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
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['training']['lr']))
    writer = SummaryWriter()

    if not os.path.exists('weights'):
        os.makedirs('weights')
    path_weights = os.path.join('weights', config['data']['dataset'] + '_best_model.pt')

    # Train the model
    if not args.only_test:

        train_acc, val_acc = train(config, model, optimizer, train_dataloader, val_dataloader, writer, save_path=path_weights, epochs=config['training']['epochs'])

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
        test_acc, test_f1 = test(config, model, test_dataloader)
        print(f'Test accuracy:{test_acc}')
        print(f'Test F1 score:{test_f1}')

    else:
        # Load the best model and test it
        model.load_state_dict(torch.load(path_weights))
        test_acc, test_f1 = test(config, model, test_dataloader)
        print(f'Test accuracy:{test_acc}')
        print(f'Test F1 score:{test_f1}')