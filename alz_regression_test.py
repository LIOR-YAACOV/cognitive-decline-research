import argparse
import os
import torch
import yaml
import numpy as np

from torch import nn
from tqdm import tqdm

from alz_sketch_data_loader import AlzData
from utils.get_model_v1 import get_model
from utils.model_utils import set_last_layer, load_ckpt
from utils.ContrastiveCenterLoss import ContrastiveCenterLoss
from torch.utils.data import DataLoader

def test_loop(model, test_dataloader, loss_fn, dev, using_center_loss=False):
    test_losses = []

    with torch.no_grad():
        if not using_center_loss:
            for images, labels, _ in tqdm(test_dataloader, desc=f"test_loop"):
                images, labels = images.to(dev), labels.to(dev)

                predictions = model(images)[-1].squeeze(1)
                current_loss = loss_fn(predictions, labels)
                test_losses.append(current_loss.item())
        else:
            for images, labels, _ in tqdm(test_dataloader, desc=f"test_loop"):
                images, labels = images.to(dev), labels.to(dev)

                feature_representation, predictions = model(images)
                predictions = predictions.squeeze(1)

                classification_loss = loss_fn[0](predictions, labels)
                center_loss = loss_fn[1](labels, feature_representation)

                current_loss = classification_loss + center_loss
                test_losses.append(current_loss.item())


    test_loss = sum(test_losses)/len(test_losses)

    torch.cuda.empty_cache()
    del images, labels

    return test_loss


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

    for i in range(1,4):
        if not os.path.isfile(config[f"test_images_list_file_{i}"]):
            print(f'file of test images list {i} not found')
            exit()
    
    for i in range(1,4):
        if not os.path.isfile(config[f"ckpt_{i}"]):
            print(f'file of ckpt of split {i} not found')
            exit()
        
    if not config["model"] is None:
        model_arch = config["model"]

    net = get_model(config)
    net, hidden_dim = set_last_layer(net, model_name=model_arch, num_classes=config["num_classes"])
    
    using_contrastive_center = config["loss_fn"] == "ContrastiveCenter"

    if config["loss_fn"] == "ContrastiveCenter":
        criterion1 = nn.CrossEntropyLoss()
        criterion2 = ContrastiveCenterLoss(hidden_dim, num_classes=config["num_classes"])

        criterion2 = criterion2.to(device)
        criterion = [criterion1, criterion2]
    else:
        criterion = nn.L1Loss()
    
    net.eval()
    
    test_set_losses = []
    
    for test_set_index in range(1,4):
        test_set = AlzData(image_list_file=config[f"test_images_list_file_{test_set_index}"],
                            size=config["size"],
                            mode="test",
                            augment_args=config['augmentations'],
                            target_label=config["target_task"])
    
        test_loader = DataLoader(test_set,
                                 batch_size=config['test_batch_size'], 
                                 shuffle=False,
                                 num_workers=config['num_workers'])

        
        ckpt_path = config[f"ckpt_{test_set_index}"]
        
        net = load_ckpt(ckpt_path=ckpt_path, model=net, device=device)
        net = net.to(device)
    
        test_loss = test_loop(net, test_loader, criterion, device, using_contrastive_center)
        
        test_set_losses.append(test_loss)
        print(f"test loss of test split {test_set_index}: {test_loss:.2f}")

    test_set_losses = np.array(test_set_losses)
    
    print(f"test accuracy :{np.mean(test_set_losses):.2f}±{np.std(test_set_losses):.2f}")
    #print(f"average test accuracy std:{np.std(test_set_losses):.2f}")