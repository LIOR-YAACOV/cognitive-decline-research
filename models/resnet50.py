import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
from .resnet18 import get_layers_list


def convert_first_layer_to_grayscale(model, model_type):
    if model_type == "ResNet":
        # print(model[0])
        # Get the first convolutional layer
        first_conv_layer = model[0]
    else:
        first_conv_layer = model.features[0][0]
    
    # Get the weights and bias of the first layer
    old_weights = first_conv_layer.weight.data
    has_bias = first_conv_layer.bias is not None
    old_bias = first_conv_layer.bias.data if has_bias else None

    # Create a new convolutional layer with 1 input channel
    new_conv = torch.nn.Conv2d(
        in_channels=1,
        out_channels=first_conv_layer.out_channels,
        kernel_size=first_conv_layer.kernel_size,
        stride=first_conv_layer.stride,
        padding=first_conv_layer.padding,
        bias=has_bias
    )

    # Calculate the weights for the new layer
    rgb_weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1).to(old_weights.device)
    new_weights = (old_weights * rgb_weights).sum(dim=1, keepdim=True)

    # Set the weights and bias for the new layer
    new_conv.weight.data = new_weights
    if has_bias:
        new_conv.bias.data = old_bias

    # Replace the first layer in the model
    if model_type == "ResNet":
        model[0] = new_conv
    else:
        model.features[0][0] = new_conv

    return model


class ResNet50v1(nn.Module):
    def __init__(self, imagenet_pretrained=False, initialize='gaussian', num_classes=250, unfreeze_option='last_layer'):
        super(ResNet50v1, self).__init__()
        if imagenet_pretrained:
            net = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')
        else:
            net = models.resnet50(weights=None)
        
        self.model = get_layers_list(net)
        
        if not imagenet_pretrained:
            self.model[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        else:
            self.model = convert_first_layer_to_grayscale(self.model, model_type="ResNet")
        
        # num_features = self.model[-1].in_features
        # self.model[-1] = nn.Linear(in_features=num_features, out_features=num_classes, bias=True)
        
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
        if self.init_method == 'kaiming':
            init.kaiming_normal_(self.model[-1].weight, mode='fan_out', nonlinearity='relu')
        elif self.init_method == 'xavier':
            init.xavier_normal_(self.model[-1].weight)
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
        for m in self.modules():
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
    resnet50v1 = ResNet50v1(imagenet_pretrained=True, unfreeze_option="all")
    test_img = torch.rand((6, 1, 512, 512))
    output1, output2 = resnet50v1(test_img)
    print(output1.shape)
    print(output2.shape)
    print(torch.argmax(output1, dim=-1))


