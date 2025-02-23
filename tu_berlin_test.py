import argparse
import os
import torch
import yaml
import numpy as np

from torch import nn
from tqdm import tqdm

from utils.get_model_v2 import get_model
from utils.model_utils import set_last_layer, load_ckpt
from utils.ContrastiveCenterLoss import ContrastiveCenterLoss
from tu_berlin_dataloader import get_test_and_folds
from torch.utils.data import DataLoader


def test_loop(model, test_dataloader, criterion, dev, using_center_loss=False):
    test_losses = []
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        if not using_center_loss:
            for images, labels in tqdm(test_dataloader, desc=f"test_loop"):
                images, labels = images.to(dev), labels.to(dev)

                predictions = model(images)
                current_loss = criterion(predictions, labels)
                test_losses.append(current_loss.item())

                predicted_labels = torch.argmax(predictions, dim=-1)
                correct_predictions += torch.sum(predicted_labels == labels).item()
                total_samples += labels.size(0)
        else:
            for images, labels in tqdm(test_dataloader, desc=f"test_loop"):
                images, labels = images.to(dev), labels.to(dev)

                feature_representation, predictions = model(images)

                classification_loss = criterion[0](predictions, labels)
                center_loss = criterion[1](labels, feature_representation)

                current_loss = classification_loss + center_loss
                test_losses.append(current_loss.item())

                predicted_labels = torch.argmax(predictions, dim=-1)
                correct_predictions += torch.sum(predicted_labels == labels).item()
                total_samples += labels.size(0)

    test_loss = sum(test_losses)/len(test_losses)
    test_accuracy = correct_predictions/total_samples

    torch.cuda.empty_cache()
    del images, labels, predictions

    return test_loss, test_accuracy


if __name__ == "__main__":
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

    ckpt_file = config["checkpoint"]

    # model_arch = ckpt_file.split("/")[2].split("_")[0]
    if not config["model"] is None:
        model_arch = config["model"]

    net = get_model(model_name=model_arch)
    net, hidden_dim = set_last_layer(net, model_name=model_arch, num_classes=config["num_classes"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # net = load_ckpt(ckpt_path=ckpt_file, model=net, device=device)
    # net = net.to(device)

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = ContrastiveCenterLoss(hidden_dim, num_classes=config["num_classes"])

    criterion2 = criterion2.to(device)
    criterion = [criterion1, criterion2]

    test_set, folds, _ = get_test_and_folds(root_dir=config["data_dir"], seed=config["seed"])

    print(len(test_set))

    test_loader = DataLoader(test_set, batch_size=config['test_batch_size'], shuffle=False,
                             num_workers=config['num_workers'])
    net.eval()
    
    test_accs = []
    
    for fold_idx, (_, _) in enumerate(folds):
        print(f"\nTest Fold {fold_idx + 1}/5")
        if fold_idx == 0:
            continue
        
        ckpt_path = ckpt_file.replace("fold0", "fold" + str(fold_idx))
        
        net = load_ckpt(ckpt_path=ckpt_path, model=net, device=device)
        net = net.to(device)
    
        test_loss, test_accuracy = test_loop(net, test_loader, criterion, device, True)
        
        test_accs.append(100*test_accuracy)
        
        print(f"test loss of fold {fold_idx + 1}: {test_loss:.2f}")
        print(f"test acc of fold {fold_idx + 1}: {100*test_accuracy:.2f}%")

    test_accs = np.array(test_accs)
    
    print(f"average test accuracy :{np.mean(test_accs):.2f}")
    print(f"average test accuracy std:{np.std(test_accs):.2f}")