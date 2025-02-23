import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
from .resnet18 import get_layers_list
from .resnet50 import convert_first_layer_to_grayscale


class ResNet152v1(nn.Module):
    def __init__(self, imagenet_pretrained=False, initialize='gaussian', unfreeze_option='last_layer'):
        super(ResNet152v1, self).__init__()
        if imagenet_pretrained:
            net = models.resnet152(weights='ResNet152_Weights.IMAGENET1K_V2')
        else:
            net = models.resnet152(weights=None)
        self.model = get_layers_list(net)

        if not imagenet_pretrained:
            self.model[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        else:
            self.model = convert_first_layer_to_grayscale(self.model, model_type="ResNet")

        self.init_method = initialize
        if imagenet_pretrained:
            self._initialize_weights()
            self._freeze_layers(unfreeze_option)
        else:
            self._initialize_all_weights()

    def forward(self, img):
        for layer in self.model[:-1]:
            img = layer(img)
        img = torch.flatten(img, 1)
        return img, self.model[-1](img)

    def _initialize_weights(self):
        # Initialize the first and last layer based on the specified method
        if self.init_method == 'kaiming':
            init.kaiming_normal_(self.model[-1].weight, mode='fan_out', nonlinearity='relu')
        elif self.init_method == 'xavier':
            init.xavier_normal_(self.model.fc.weight)
        elif self.init_method == 'gaussian':
            init.normal_(self.model[-1].weight, mean=0, std=0.01)

    def _freeze_layers(self, option):
        # Freeze all layers first
        for param in self.parameters():
            param.requires_grad = False

        if option == 'last_layer':
            # Unfreeze only the last layer (original behavior)
            for param in self.model[-1].parameters():
                param.requires_grad = True

        elif option == 'last_block':
            # Unfreeze the last residual block and fully connected layer
            for param in self.model[-3:].parameters():
                param.requires_grad = True

        elif option == 'last_three_blocks':
            # Unfreeze the last three residual blocks and fully connected layer
            for param in self.model[-7:].parameters():
                param.requires_grad = True

        elif option == 'all':
            # Unfreeze all layers
            for param in self.parameters():
                param.requires_grad = True

    def _initialize_all_weights(self):
        # Initialize weights for all layers based on the specified method
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if self.init_method == 'kaiming':
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif self.init_method == 'xavier':
                    init.xavier_normal_(m.weight)
                elif self.init_method == 'gaussian':
                    init.normal_(m.weight, mean=0, std=0.01)
            if isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


if __name__ == '__main__':
    net = ResNet152v1()
    # print(net)
    # print([p.numel() for p in net.parameters()])
    test_img = torch.rand((6, 1, 512, 512))
    output = net(test_img)
    print(output.shape)
