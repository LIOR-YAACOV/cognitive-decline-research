import torch
import argparse
import os
import yaml

import numpy as np

from tqdm import tqdm
from torch.optim import Adam, AdamW
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch import nn

from alz_sketch_data_loader import AlzData
from utils.get_model_v1 import get_model
from utils.model_utils import set_last_layer
from utils.lr_utils import LearningRateWarmUP
from utils.plot_results import plot_results


def train_loop(model, train_dl, loss_func, optim, curr_device, epoch_num):
    epoch_losses = []

    print(f"Model is on device: {next(model.parameters()).device}")

    model.train()
    for images, labels, _ in tqdm(train_dl, desc=f"train_loop , epoch {epoch_num}"):
        images, labels = images.to(curr_device), labels.to(curr_device)

        optim.zero_grad()
        predictions = model(images)[-1]
        #print(predictions)
        labels = labels.to(torch.float64)
        predictions = predictions.to(torch.float64).squeeze(1)
        current_loss = loss_func(predictions, labels)
        current_loss.backward()
        optim.step()

        epoch_losses.append(current_loss.item())

    epoch_loss = sum(epoch_losses) / len(epoch_losses)
    torch.cuda.empty_cache()

    del images, labels  # , predictions

    return epoch_loss  # , epoch_accuracy


def val_loop(model, val_dl, loss_func, curr_device, epoch_num):
    epoch_losses = []

    print(f"Model is on device: {next(model.parameters()).device}")

    model.eval()
    with torch.no_grad():
        for images, labels, _ in tqdm(val_dl, desc=f"validation_loop , epoch {epoch_num}"):
            images, labels = images.to(curr_device), labels.to(curr_device)

            predictions = model(images)[-1]
            labels = labels.to(torch.float64)
            predictions = predictions.to(torch.float64).squeeze(1)
            current_loss = loss_func(predictions, labels)
            epoch_losses.append(current_loss.item())

    epoch_loss = sum(epoch_losses) / len(epoch_losses)
    del images, labels  # , predictions

    return epoch_loss  # , epoch_accuracy


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', nargs="?", type=str, default="configs/blabla.yml",
                        help="Configuration file to use")

    args = parser.parse_args()

    if not os.path.isfile(args.config_file):
        print('Configuration file not found')
        exit()

    with open(args.config_file) as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    print(f"config file is {args.config_file}")

    if not os.path.isfile(config["train_images_list_file"]):
        print('file of train images list not found')
        exit()

    if not os.path.isfile(config["val_images_list_file"]):
        print('file of validation images list not found')
        exit()

    torch.backends.cudnn.benchmark = True

    alzheimer_train_dataset = AlzData(image_list_file=config["train_images_list_file"],
                                       size=config["size"],
                                       mode="train",
                                       augment_args=config['augmentations'],
                                       target_label=config["target_task"])

    alzheimer_val_dataset = AlzData(image_list_file=config["val_images_list_file"],
                                     size=config["size"],
                                     mode="val",
                                     target_label=config["target_task"])

    train_dataloader = DataLoader(alzheimer_train_dataset, batch_size=config['train_batch_size'],
                                  shuffle=True, num_workers=config['num_workers'])

    validation_dataloader = DataLoader(alzheimer_val_dataset, batch_size=config['val_batch_size'],
                                       shuffle=True, num_workers=config['num_workers'])

    print(f"train set size = {len(alzheimer_train_dataset)}")
    print(f"validation set size = {len(alzheimer_val_dataset)}")

    best_train_loss = np.Inf
    best_val_loss = np.Inf

    train_losses = []
    validation_losses = []

    net = get_model(configure=config)

    start_lr = config['lr']

    curr_epoch = 1
    num_epochs = config['num_epochs']

    warm_up_epochs = 0

    if config["optimizer"] == "Adam":
        optimizer = Adam(net.parameters(), lr=start_lr, weight_decay=config['weight_decay'])
    else:
        optimizer = AdamW(net.parameters(), lr=start_lr, weight_decay=config['weight_decay'])

    # using_warmup = False

    num_classes = config["num_classes"]
    tu_berlin_ckpt = config["tu_berlin_ckpt"]

    if tu_berlin_ckpt is not None and Path(tu_berlin_ckpt).is_file():
        saved_state = torch.load(tu_berlin_ckpt, map_location=device)
        net, _ = set_last_layer(net, config["model"], num_classes=250)
        net.load_state_dict({k: v.to(device) for k, v in saved_state['model_state'].items()})

        print(f"tu berlin pretrain ckpt {tu_berlin_ckpt} loaded")
        print(f"using tu-berlin dataset pretrained network")

        net, _ = set_last_layer(net, config["model"], num_classes)
        net = net.to(device)

        warm_up_epochs = config['warm_up_epochs']
        print(f"warm_up_epochs =  {warm_up_epochs}")

        after_warm_up_sched = ReduceLROnPlateau(optimizer, patience=config['lr_patience'],
                                                factor=config['lr_decay_factor'], mode='min')
        lr_sched = LearningRateWarmUP(optimizer, warm_up_epochs, start_lr,
                                      after_scheduler=after_warm_up_sched)

    else:
        print(f"not using transfer learning")

        net, _ = set_last_layer(net, config["model"], num_classes)
        net = net.to(device)

        after_warm_up_sched = ReduceLROnPlateau(optimizer, patience=config['lr_patience'],
                                                factor=config['lr_decay_factor'], mode='min')
        warm_up_epochs = config['warm_up_epochs']
        lr_sched = LearningRateWarmUP(optimizer, warm_up_epochs, start_lr,
                                      after_scheduler=after_warm_up_sched)

    loss_fn = config["loss_function"]

    if loss_fn == "L2":
        loss_fn = nn.MSELoss()
    elif loss_fn == "L1":
        loss_fn = nn.L1Loss()

    for epoch in range(1, config['num_epochs']+1):
        print(f"current epoch = {epoch}")

        train_loss = train_loop(net, train_dataloader, loss_fn, optimizer, device, epoch)

        train_losses.append(train_loss)
        print(f"current train loss: {train_loss}")

        val_loss = val_loop(net, validation_dataloader, loss_fn,  device, epoch)
        print(f"current validation loss: {val_loss}")
        validation_losses.append(val_loss)

        if epoch < warm_up_epochs+1:
            lr_sched.step()
        else:
            lr_sched.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if train_loss < best_train_loss:
                best_train_loss = train_loss
            if config["best_ckpt"]:
                torch.save({
                    'epoch': epoch,
                    'model': net.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': validation_losses,
                    'best_val_loss': best_val_loss
                }, config["best_ckpt"])
                print(f"saved checkpoint" + config["best_ckpt"])

    if tu_berlin_ckpt:
        plot_results(train_losses, validation_losses, config=config)
    else:
        plot_results(train_losses, validation_losses, config=config)

    print(f"best train loss: {best_train_loss}")
    print(f"best validation loss: {best_val_loss}")
