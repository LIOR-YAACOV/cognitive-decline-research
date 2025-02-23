import yaml
import torch
import argparse
import numpy as np
import os

from collections import defaultdict
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from tu_berlin_dataloader import get_test_and_folds, SketchData
from utils.lr_utils import LearningRateWarmUP
from utils.get_model_v1 import get_model
from utils.model_utils import set_last_layer
from utils.ContrastiveCenterLoss import ContrastiveCenterLoss

def train_loop(model, train_dl, criterion, optimizer, device, epoch_num, optim_center=None, using_center_loss=False):
    epoch_losses = []

    correct_predictions = 0
    total_samples = 0

    model.train()
    if not using_center_loss:
        for imgs, labels in tqdm(train_dl, desc=f"train_loop , epoch {epoch_num}"):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            _, predictions = model(imgs)
            current_loss = criterion(predictions, labels)
            current_loss.backward()
            optimizer.step()

            epoch_losses.append(current_loss.item())

            predicted_labels = torch.argmax(predictions, dim=-1)
            correct_predictions += torch.sum(predicted_labels == labels).item()

            total_samples += labels.size(0)

        # epoch_loss = sum(epoch_losses) / len(epoch_losses)
        # epoch_accuracy = correct_predictions / total_samples
    else:
        for imgs, labels in tqdm(train_dl, desc=f"train_loop , epoch {epoch_num}"):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            optim_center.zero_grad()

            feature_representation, predictions = model(imgs)

            classification_loss = criterion[0](predictions, labels)
            center_loss = criterion[1](labels, feature_representation)

            current_loss = classification_loss + center_loss

            current_loss.backward()

            optimizer.step()
            optim_center.step()

            epoch_losses.append(current_loss.item())

            predicted_labels = torch.argmax(predictions, dim=-1)
            correct_predictions += torch.sum(predicted_labels == labels).item()

            total_samples += labels.size(0)

    epoch_loss = sum(epoch_losses) / len(epoch_losses)
    epoch_accuracy = correct_predictions / total_samples

    torch.cuda.empty_cache()
    del imgs, labels, predictions

    return epoch_loss, epoch_accuracy


def val_loop(model, val_dl, criterion, device, epoch_num, using_center_loss=False):
    epoch_losses = []

    correct_predictions = 0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        if not using_center_loss:
            for images, labels in tqdm(val_dl, desc=f"validation_loop , epoch {epoch_num}"):
                images, labels = images.to(device), labels.to(device)

                predictions = model(images)
                current_loss = criterion(predictions, labels)
                epoch_losses.append(current_loss.item())

                predicted_labels = torch.argmax(predictions, dim=-1)
                correct_predictions += torch.sum(predicted_labels == labels).item()
                total_samples += labels.size(0)
        else:
            for images, labels in tqdm(val_dl, desc=f"validation_loop , epoch {epoch_num}"):
                images, labels = images.to(device), labels.to(device)

                feature_representation, predictions = model(images)

                classification_loss = criterion[0](predictions, labels)
                center_loss = criterion[1](labels, feature_representation)

                current_loss = classification_loss + center_loss
                epoch_losses.append(current_loss.item())

                predicted_labels = torch.argmax(predictions, dim=-1)
                correct_predictions += torch.sum(predicted_labels == labels).item()
                total_samples += labels.size(0)

    epoch_loss = sum(epoch_losses) / len(epoch_losses)
    epoch_accuracy = correct_predictions / total_samples
    del images, labels, predictions

    return epoch_loss, epoch_accuracy


def train_k_fold(config, device):
    path_to_data = config['data_dir']
    test_set, folds, train_val_indices = get_test_and_folds(root_dir=path_to_data, seed= config['seed'])

    # Track metrics across folds
    fold_metrics = defaultdict(list)

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\nTraining Fold {fold_idx + 1}/5")
        
        if fold_idx > 0:
            continue
            
        # Map fold indices back to original dataset indices
        train_indices = train_val_indices[train_idx]
        val_indices = train_val_indices[val_idx]

        # Create datasets for this fold
        train_set = SketchData(image_indices=train_indices, path_to_data=path_to_data,
                                mode='train', invert_pixels=config["invert_colors"])
        val_set = SketchData(image_indices=val_indices, path_to_data=path_to_data,
                              mode='val', invert_pixels=False)

        # Create dataloaders
        train_loader = DataLoader(train_set, batch_size=config['train_batch_size'],
                                  shuffle=True, num_workers=config['num_workers'])
        val_loader = DataLoader(val_set, batch_size=config['val_batch_size'],
                                shuffle=False, num_workers=config['num_workers'])

        # Initialize model for this fold
        model = get_model(configure=config)
        model, hidden_dim = set_last_layer(model, config["model"], config["num_classes"])
        print(f"set last layer of model to {config['num_classes']} ")

        model = model.to(device)
        print(f"model was moved to device = {device}")
        using_center_loss = False
        optimizer_center = None

        # Initialize optimizer and criterion
        if config["loss_function"] == "CrossEntropy":
            criterion = nn.CrossEntropyLoss()
        elif config["loss_function"] == "L1":
            criterion = nn.L1Loss()
        elif config["loss_function"] == "ContrastiveCenter":
            criterion1 = nn.CrossEntropyLoss()
            criterion2 = ContrastiveCenterLoss(hidden_dim,
                                               num_classes=config["num_classes"])
            criterion2 = criterion2.to(device)
            optimizer_center = optim.SGD(criterion2.parameters(),
                                         lr=config["center_loss_lr"])
            criterion = [criterion1, criterion2]
            using_center_loss = True
        else:
            criterion = nn.MSELoss()

        if config["optimizer"] == "Adam":
            optimizer = Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        else:
            optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

        # Initialize learning rate scheduler
        if config.get('warm_up_epochs', 0) > 0:
            after_scheduler = ReduceLROnPlateau(optimizer, 
                                                patience=config['lr_patience'],
                                                factor=config['lr_decay_factor'], 
                                                mode='max')
                                                
            lr_scheduler = LearningRateWarmUP(optimizer, 
                                              warm_up_epochs=config['warm_up_epochs'],
                                              target_lr=optimizer.param_groups[0]['lr'], 
                                              after_scheduler=after_scheduler)
            using_warmup = True
        else:
            lr_scheduler = ReduceLROnPlateau(optimizer, 
                                            patience=config['lr_patience'],
                                            factor=config['lr_decay_factor'], 
                                            mode='max')
            using_warmup = False

        # Training loop for this fold
        best_val_acc = 0
        epoch_of_best_acc = 1
        best_train_acc = 0
        train_epochs_losses = []
        val_epoch_losses = []
        train_epochs_accuracy = []
        val_epoch_accuracy = []

        for epoch in range(1, config['num_epochs']+1):
            print(f"Fold {fold_idx + 1}, Epoch {epoch}")

            train_loss, train_acc = train_loop(model, train_loader, criterion, optimizer,
                                               device, epoch, optim_center=optimizer_center,
                                               using_center_loss=using_center_loss)
            val_loss, val_acc = val_loop(model, val_loader, criterion, device, epoch,
                                         using_center_loss=using_center_loss)

            # Store metrics
            train_epochs_losses.append(train_loss)
            train_epochs_accuracy.append(train_acc)
            val_epoch_losses.append(val_loss)
            val_epoch_accuracy.append(val_acc)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Update learning rate
            if using_warmup and epoch < config['warm_up_epochs']:
                lr_scheduler.step()
            else:
                lr_scheduler.step(val_acc)

            # Save best model for this fold
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epoch_of_best_acc = epoch
                if train_acc > best_train_acc:
                    best_train_acc = train_acc
                if config["best_ckpt_path"]:
                    torch.save({
                        'fold': fold_idx,
                        'epoch': epoch,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'train_losses': train_epochs_losses,
                        'train_accuracies': train_epochs_accuracy,
                        'val_losses': val_epoch_losses,
                        'val_accuracies': val_epoch_accuracy,
                        'best_val_accuracy': best_val_acc
                    }, f"{config['best_ckpt_path']}_fold{fold_idx}.pth")
                    print(f"saved {config['best_ckpt_path']}_fold{fold_idx}.pth")

        # Store fold metrics
        fold_metrics[f"fold_index_{fold_idx}"].append((best_val_acc, epoch_of_best_acc, best_train_acc))

        # Clean up
        del model, optimizer
        torch.cuda.empty_cache()
        # break

    val_accuracies = []
    train_accuracies = []
    # Print cross-validation results
    for fold_idx in fold_metrics.keys():
        val_accuracies.append(fold_metrics[fold_idx][0][0])
        train_accuracies.append(fold_metrics[fold_idx][0][-1])

    mean_train_acc = np.mean(train_accuracies)
    std_train_acc = np.std(train_accuracies)
    
    mean_val_acc = np.mean(val_accuracies)
    std_val_acc = np.std(val_accuracies)
    print(f"\nCross-validation results:")
    print(f"Mean train accuracy: {100*mean_train_acc:.4f} ± {std_train_acc:.4f}")
    print(f"Mean validation accuracy: {100*mean_val_acc:.4f} ± {std_val_acc:.4f}")

    return fold_metrics, test_set


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default="configs/blabla.yml")
    args = parser.parse_args()

    if not os.path.isfile(args.config_file):
        print('Configuration file not found')
        exit()

    with open(args.config_file) as fp:
        config_file = yaml.load(fp, Loader=yaml.FullLoader)

    if not config_file["best_ckpt_path"]:
        print('no file for best ckpt save')
        exit()

    if not config_file["ckpt_path"]:
        print('no file for final ckpt save')
        exit()

    print(f"config file = {args.config_file}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics, _ = train_k_fold(config_file, dev)
