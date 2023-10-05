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


def train(num_classes, model, optimizer, train_dataloader, val_dataloader, writer, save_path= 'best_model.pt', epochs=10):
    criterion = torch.nn.CrossEntropyLoss()
    best_val_acc = 0
    train_acc_macro_list = []
    val_acc_macro_list = []
    train_acc_micro_list = []
    val_acc_micro_list = []
    train_acc_macro = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average='macro').to(device)
    val_acc_macro = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average='macro').to(device)
    train_acc_micro = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average='micro').to(device)
    val_acc_micro = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average='micro').to(device)
    for epoch in range(epochs):
        print(f'Epoch {epoch}/{epochs} :')
        running_loss = []
        train_data = tqdm(train_dataloader)
        for features, labels in train_data:
            features, labels = features.to(device), labels.to(device)
            # Forward pass
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            # Metrics
            train_acc_macro(preds, labels)
            train_acc_micro(preds, labels)
            loss = criterion(outputs, labels)
            running_loss.append(loss.item())
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_data.set_description(f'training loss: {mean(running_loss)}')
        # Training accuracy of the epoch
        writer.add_scalar('training loss', mean(running_loss), epochs)
        total_train_acc_macro = train_acc_macro.compute().cpu().data.numpy()
        total_train_acc_micro = train_acc_micro.compute().cpu().data.numpy()
        print(f'Training accuracy macro:{total_train_acc_macro}')
        train_acc_macro_list.append(total_train_acc_macro)
        print(f'Training accuracy micro:{total_train_acc_micro}')
        train_acc_micro_list.append(total_train_acc_micro)

        # Validation step 
        for features, labels in val_dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            val_acc_macro(preds, labels)
            val_acc_micro(preds, labels)
        
        total_val_acc_macro = val_acc_macro.compute().cpu().data.numpy()
        total_val_acc_micro = val_acc_micro.compute().cpu().data.numpy()
        print(f'Validation accuracy macro:{total_val_acc_macro}')
        val_acc_macro_list.append(total_val_acc_macro)
        print(f'Validation accuracy micro:{total_val_acc_micro}')
        val_acc_micro_list.append(total_val_acc_micro)

        if total_val_acc_macro > best_val_acc:
            best_val_acc = total_val_acc_macro
            torch.save(model.state_dict(), os.path.join(save_path))

    return train_acc_macro_list, train_acc_micro_list, val_acc_macro_list, val_acc_micro_list

def test(num_classes, model, test_dataloader):
    acc_by_class = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average=None).to(device)
    test_acc_macro = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average='macro').to(device)
    test_acc_micro = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average='micro').to(device)
    f1_by_class = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average=None).to(device)
    f1_score_macro = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='macro').to(device)
    f1_score_micro = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='micro').to(device)
    with torch.no_grad():
        for features, labels in test_dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            # Metrics
            test_acc_macro(preds, labels)
            test_acc_micro(preds, labels)
            f1_score_macro(preds, labels)
            f1_score_micro(preds, labels)
            # Metrics by class 
            acc_by_class(preds, labels)
            f1_by_class(preds, labels)

        total_test_acc_macro = test_acc_macro.compute().cpu().data.numpy()
        total_test_acc_micro = test_acc_micro.compute().cpu().data.numpy()
        total_f1_score_macro = f1_score_macro.compute().cpu().data.numpy()
        total_f1_score_micro = f1_score_micro.compute().cpu().data.numpy()
        total_acc_by_class = acc_by_class.compute().cpu().data.numpy() 
        total_f1_by_class = f1_by_class.compute().cpu().data.numpy()
        print(f'Test accuracy macro:{total_test_acc_macro}')
        print(f'Test accuracy micro:{total_test_acc_micro}')
        print(f'Test F1 score macro:{total_f1_score_macro}')
        print(f'Test F1 score micro:{total_f1_score_micro}')
        print(f'Accuracy by class:{total_acc_by_class}')
        print(f'F1 score by class:{total_f1_by_class}')

    #return total_test_acc_macro, total_test_acc_micro, total_f1_score_macro, total_f1_score_micro, total_acc_by_class, total_f1_by_class

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

        train_acc_macro, train_acc_micro, val_acc_macro, val_acc_micro = train(num_classes, 
                                                                               model, 
                                                                               optimizer, 
                                                                               train_dataloader, 
                                                                               val_dataloader, 
                                                                               writer, 
                                                                               save_path=path_weights, 
                                                                               epochs=config['training']['epochs']
                                                                               )

        # Plot the training and validation accuracy and save it 
        plt.plot(train_acc_macro, color='b', label='Training accuracy macro')
        plt.plot(train_acc_micro, color='g', label='Training accuracy micro')
        plt.plot(val_acc_macro, color='r', label='Validation accuracy macro')
        plt.plot(val_acc_micro, color='y', label='Validation accuracy micro')
        plt.legend()
        plt.show()
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.savefig(os.path.join('plots', config['data']['dataset'] + '_accuracy.png'))
        plt.close()

    # Load the best model and test it
    model.load_state_dict(torch.load(path_weights))
    # test_acc_macro, test_acc_micro, test_f1_macro, test_f1_micro, test_acc_by_class, test_f1_by_class = test(num_classes,
    #                                                                                                          model, 
    #                                                                                                          test_dataloader
    #                                                                                                          )
    # Get the statistics of the best model on training set 
    test(num_classes, model, train_dataloader)
    # Get the statistics of the best model on validation set
    test(num_classes, model, val_dataloader)
    # Get the statistics of the best model on test set
    test(num_classes, model, test_dataloader)