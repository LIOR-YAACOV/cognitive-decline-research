import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
# ResNetTypes = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']


def get_layers_list(network):
    layers = []
    for name, block in network.named_children():
        layers.append(block)
    return nn.Sequential(*layers)


class ResNet18v1(nn.Module):
    def __init__(self, imagenet_pretrained=False, initialize='gaussian'):
        super(ResNet18v1, self).__init__()
        if imagenet_pretrained:
            net = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
        else:
            net = models.resnet18(weights=None)
        self.model = get_layers_list(net)

        self.model[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # num_features = self.model[-1].in_features
        # self.model[-1] = nn.Linear(in_features=num_features, out_features=250, bias=True)
        self.init_method = initialize
        if imagenet_pretrained:
            self._initialize_weights()
            self._freeze_layers()
        else:
            self._initialize_all_weights()

    def forward(self, img):
        for layer in self.model[:-1]:
            img = layer(img)
        img = torch.flatten(img, 1)
        return self.model[-1](img), img

    def _initialize_weights(self):
        # Initialize the first and last layer based on the specified method
        if self.init_method == 'kaiming':
            init.kaiming_normal_(self.model[0].weight, mode='fan_out', nonlinearity='relu')
            init.kaiming_normal_(self.model[-1].weight, mode='fan_out', nonlinearity='relu')
        elif self.init_method == 'xavier':
            init.xavier_normal_(self.model[0].weight)
            init.xavier_normal_(self.model.fc.weight)
        elif self.init_method == 'gaussian':
            init.normal_(self.model[0].weight, mean=0, std=0.01)
            init.normal_(self.model[-1].weight, mean=0, std=0.01)

    def _freeze_layers(self):
        # Freeze all layers first
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze the first and last layers
        for param in self.model[0].parameters():
            param.requires_grad = True
        for param in self.model[-1].parameters():
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
    resnet18v1 = ResNet18v1()
    test_img = torch.rand((1, 1, 512, 512))
    output = resnet18v1(test_img)
    print(output.shape)
