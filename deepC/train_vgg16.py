import argparse
import yaml
from tqdm import tqdm

from statistics import mean

from dataset import VehicleDataset

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

def vgg16(nb_classes):
    # Load VGG16 model 
    vgg16 = torchvision.models.get_model('vgg16', weights=None)
    # Change the input layer
    vgg16.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # Modify the last layer
    vgg16.classifier[6] = torch.nn.Linear(4096, nb_classes)
    return vgg16

def train(model, optimizer, loader, writer, epochs=10):
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = []
        t = tqdm(loader)
        for x, y in t:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            running_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(f'training loss: {mean(running_loss)}')
        writer.add_scalar('training loss', mean(running_loss), epochs)

def test(model, dataloader):
    test_corrects = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x).argmax(1)
            test_corrects += y_hat.eq(y).sum().item()
            total += y.size(0)
    return test_corrects / total

if __name__ == "__main__": 

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='conf.yaml', help='Path to config file.')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataloader = DataLoader(VehicleDataset(config), batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'])
    val_dataloader = DataLoader(VehicleDataset(config, set='val'), batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'])
    test_dataloader = DataLoader(VehicleDataset(config, set='test'), batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'])

    model = vgg16(len(config['data']['classes'])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
    writer = SummaryWriter()

    train(model, optimizer, train_dataloader, writer, epochs=config['training']['epochs'])

    test_acc = test(vgg16, test_dataloader)
    print(f'Test accuracy:{test_acc}')