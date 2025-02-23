from matplotlib import pyplot as plt
import torch
import argparse
import os

import numpy as np

def plot_results(train_losses, validation_losses, config, train_accuracies=None, val_accuracies=None, fold_index=None):
    train_loss_indices = list(range(len(train_losses)))
    val_loss_indices = list(range(len(validation_losses)))
    best_ckpt = config["best_ckpt"].split(".")[0]

    model = config['model']
    used_augments = "aug" in best_ckpt
    
    if config['target'] == 'classification':
        #Tu Berlin classification
        if train_accuracies and val_accuracies:
            train_acc_indices = list(range(len(train_accuracies)))
            val_acc_indices = list(range(len(val_accuracies)))
            seed = config["seed"]

            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
            ax[0].plot(train_loss_indices, train_losses, label="train_loss")
            ax[0].plot(val_loss_indices, validation_losses, label="val_loss")
            ax[0].legend()
            ax[1].plot(train_acc_indices, train_accuracies, label="train_acc")
            ax[1].plot(val_acc_indices, val_accuracies, label="val_acc")
            ax[1].legend()

            plt.suptitle(model + f"{fold_index}")
            if not os.path.exists(os.path.join(os.getcwd(), "model_scores")):
                os.makedirs(os.path.join(os.getcwd(), "model_scores"))

            data_split_index = best_ckpt.split("_")[-3]

            if used_augments:
                save_path = os.path.join(os.getcwd(), "model_scores", seed, model + "_aug_split" + data_split_index)
            else:
                save_path = os.path.join(os.getcwd(), "model_scores", seed, model + "split" + data_split_index)
            plt.savefig(save_path, format='png')
    else:
        #target is regression - collected alzheimer dataset
        fig = plt.figure(figsize=(15, 5))
        plt.plot(train_loss_indices, train_losses, label="train_loss")
        plt.plot(val_loss_indices, validation_losses, label="val_loss")
        fig.legend()

        plt.title(model)
        if not os.path.exists(os.path.join(os.getcwd(), "model_scores")):
            os.makedirs(os.path.join(os.getcwd(), "model_scores"))
        
        img_name = best_ckpt.split("/")[1]
        save_path = os.path.join(os.getcwd(), "model_scores", img_name)
        plt.savefig(save_path)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt', nargs="?", type=str, default="new_saved_models/seed_42/blabla.pth",
                        help="Configuration file to use")

    args = parser.parse_args()

    # if not os.path.isfile(args.ckpt):
    #     print('Ckpt file not found')
    #     exit()

    device = "cpu"

    base_ckpt = args.ckpt

    current_seed, model_name = base_ckpt.split("/")[-2], base_ckpt.split("/")[-1]
    num_folds = 5

    train_accs = []
    val_accs = []

    for fold_idx in range(1, num_folds):
        ckpt = base_ckpt + f"{fold_idx}.pth"
        saved_state = torch.load(ckpt, map_location=device)
        print(f"current mode is {model_name}{fold_idx}.pth")

        train_fold_losses = saved_state['train_losses']
        val_fold_losses = saved_state['val_losses']
        train_fold_accs = saved_state['train_accuracies']
        val_fold_accs = saved_state['val_accuracies']

        max_accuracy_idx = np.argmax(np.array(val_fold_accs))

        train_accs.append(100*train_fold_accs[max_accuracy_idx])
        val_accs.append(100*val_fold_accs[max_accuracy_idx])
        # print(f"train acc: {100*train_acc[idx]:.2f}%")
        # print(f"max validation acc: {100*val_acc[idx]:.2f}%")
        plot_results(train_fold_losses, val_fold_losses, model_name, current_seed,train_fold_accs, val_fold_accs, fold_idx)

    print(f"train accuracy : {np.mean(train_accs):.2f} ± {np.std(train_accs):.4f}")
    print(f"validation accuracy : {np.mean(val_accs):.2f} ± {np.std(val_accs)/100:.4f}")
