import argparse
import os
import yaml
import pickle
from tqdm import tqdm

from statistics import mean

from dataset import VehicleDataset

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchsummary import summary

def vgg16(nb_classes):
    # Load VGG16 model 
    vgg16 = torchvision.models.get_model('vgg16', weights=None)
    # Change the input layer
    vgg16.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # Modify the last layer
    vgg16.classifier[6] = torch.nn.Linear(4096, nb_classes)
    return vgg16

def vgg11(nb_classes):
    # Load VGG16 model 
    vgg11 = torchvision.models.get_model('vgg11', weights=None)
    # Change the input layer
    vgg11.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # Modify the last layer
    vgg11.classifier[6] = torch.nn.Linear(4096, nb_classes)
    return vgg11

def train(model, optimizer, loader, writer, epochs=10):
    criterion = torch.nn.CrossEntropyLoss()
    corrects = 0
    total = 0
    best_val_acc = 0
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
        writer.add_scalar('training loss', mean(running_loss), epochs)
        writer.add_scalar('training accuracy', accuracy, epochs)
        print(f'Training accuracy:{accuracy}')

        # Validation step 
        val_acc = test(model, val_dataloader)
        print(f'Validation accuracy:{val_acc}')
        writer.add_scalar('validation accuracy', val_acc, epochs)
        writer.flush()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')

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
    
    train_dataloader = DataLoader(VehicleDataset(partition, config), batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'])
    val_dataloader = DataLoader(VehicleDataset(partition, config, set='val'), batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'])
    test_dataloader = DataLoader(VehicleDataset(partition, config, set='test'), batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'])
    
    model = vgg11(len(config['data']['classes'])).to(device)
    print(model)
    print(summary(model, train_dataloader.dataset[0][0].shape))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
    writer = SummaryWriter()

    train(model, optimizer, train_dataloader, writer, epochs=config['training']['epochs'])

    # Load the best model
    model.load_state_dict(torch.load('best_model.pt'))

    test_acc = test(model, test_dataloader)
    print(f'Test accuracy:{test_acc}')