import torch
from torch import nn

def save_ckpt(ckpt_path, fold_idx, epoch, model, optimizer, train_losses,
              train_accs, val_losses, val_accs, best_val_acc):
    torch.save({
                        'fold': fold_idx,
                        'epoch': epoch,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'train_losses': train_losses,
                        'train_accuracies': train_accs,
                        'val_losses': val_losses,
                        'val_accuracies': val_accs,
                        'best_val_accuracy': best_val_acc
                    }, ckpt_path)
    print(f"saved ckpt {ckpt_path}")


def load_ckpt(ckpt_path, model, device):
    saved_state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(saved_state["model"])
    print(f"loaded checkpoint {ckpt_path}")
    return model


def set_last_layer(model, model_name, num_classes):
    if "resnet" in model_name:
        hidden_dim = model.model[-1].in_features
        model.model[-1] = nn.Linear(in_features=hidden_dim, out_features=num_classes, bias=True)

    else:  # convnext
        hidden_dim = model.model[-1][-1].in_features
        model.model[-1][-1] = nn.Linear(in_features=hidden_dim, out_features=num_classes, bias=True)

    return model, hidden_dim


def get_loss_function(loss_func_name):
    if loss_func_name == "CrossEntropy":
        return nn.CrossEntropyLoss()
    elif loss_func_name == "L1":
        return nn.L1Loss()
    else:
        return nn.MSELoss()
