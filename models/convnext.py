import torch

from torch import nn
from torchvision.models import convnext_tiny, convnext_small, convnext_base, convnext_large

from .resnet50 import convert_first_layer_to_grayscale
from .resnet18 import get_layers_list


class ConvNextTiny(nn.Module):
    def __init__(self, pretrained=None, unfreeze_option='last_layer', num_classes=1):
        super(ConvNextTiny, self).__init__()
        if pretrained:
            model = convnext_tiny(weights='ConvNeXt_Tiny_Weights.IMAGENET1K_V1')
        else:
            model = convnext_tiny(weights=None)

        if not pretrained:
            model.features[0][0] = nn.Conv2d(in_channels=1, out_channels=model.features[0][0].out_channels,
                                             kernel_size=4, stride=4)
        else:
            model = convert_first_layer_to_grayscale(model, model_type="ConvNext")

        model_blocks = get_layers_list(model)

        self.model = model_blocks

        if pretrained:
            self._freeze_layers(unfreeze_option)

    def _freeze_layers(self, option):
        # Freeze all layers first
        for param in self.model.parameters():
            param.requires_grad = False
        
        if option == 'last_layer':
            # Unfreeze only the last layer (original behavior)
            for param in self.model[-1][-1].parameters():
                param.requires_grad = True
        
        elif option == 'classifier':
            # Unfreeze the entire classifier
            for param in self.model[-1].parameters():
                param.requires_grad = True
        
        elif option == 'last_block':
            # Unfreeze the last convolutional block and classifier
            for param in self.model[0][-1].parameters():
                param.requires_grad = True
            for param in self.model[-1].parameters():
                param.requires_grad = True
        
        elif option == 'last_three_blocks':
            # Unfreeze the last three convolutional blocks and classifier
            for i in range(-3, 0):
                for param in self.model[0].parameters():
                    param.requires_grad = True
            for param in self.model[-1].parameters():
                param.requires_grad = True
        
        elif option == 'all':
            # Unfreeze all layers
            for param in self.model.parameters():
                param.requires_grad = True

    def forward(self, x):
        for block in self.model[:-1]:
            x = block(x)
        for layer in self.model[-1][:-1]:
            x = layer(x)

        return x, self.model[-1][-1](x)


class ConvNextSmall(nn.Module):
    def __init__(self, pretrained=None, unfreeze_option='last_layer', num_classes=1):
        super(ConvNextSmall, self).__init__()
        if pretrained:
            model = convnext_small(weights='ConvNeXt_Small_Weights.IMAGENET1K_V1')
        else:
            model = convnext_small(weights=None)
        if not pretrained:
            model.features[0][0] = nn.Conv2d(in_channels=1, out_channels=model.features[0][0].out_channels,
                                             kernel_size=4, stride=4)
        else:
            model = convert_first_layer_to_grayscale(model, model_type="ConvNext")

        model_blocks = get_layers_list(model)

        self.model = model_blocks

        if pretrained:
            self._freeze_layers(unfreeze_option)

    def _freeze_layers(self, option):
        # Freeze all layers first
        for param in self.model.parameters():
            param.requires_grad = False

        if option == 'last_layer':
            # Unfreeze only the last layer (original behavior)
            for param in self.model[-1][-1].parameters():
                param.requires_grad = True

        elif option == 'classifier':
            # Unfreeze the entire classifier
            for param in self.model[-1].parameters():
                param.requires_grad = True

        elif option == 'last_block':
            # Unfreeze the last convolutional block and classifier
            for param in self.model[0][-1].parameters():
                param.requires_grad = True
            for param in self.model[-1].parameters():
                param.requires_grad = True

        elif option == 'last_three_blocks':
            # Unfreeze the last three convolutional blocks and classifier
            for i in range(-3, 0):
                for param in self.model[0].parameters():
                    param.requires_grad = True
            for param in self.model[-1].parameters():
                param.requires_grad = True

        elif option == 'all':
            # Unfreeze all layers
            for param in self.model.parameters():
                param.requires_grad = True

    def forward(self, x):
        for block in self.model[:-1]:
            x = block(x)
        for layer in self.model[-1][:-1]:
            x = layer(x)

        return x, self.model[-1][-1](x)


class ConvNextBase(nn.Module):
    def __init__(self, pretrained=None, unfreeze_option='last_layer', num_classes=1):
        super(ConvNextBase, self).__init__()
        if pretrained:
            model = convnext_base(weights='ConvNeXt_Base_Weights.IMAGENET1K_V1')
        else:
            model = convnext_base(weights=None)

        if not pretrained:
            model.features[0][0] = nn.Conv2d(in_channels=1, out_channels=model.features[0][0].out_channels,
                                             kernel_size=4, stride=4)
        else:
            model = convert_first_layer_to_grayscale(model, model_type="ConvNext")

        model_blocks = get_layers_list(model)

        self.model = model_blocks

        if pretrained:
            self._freeze_layers(unfreeze_option)

    def _freeze_layers(self, option):
        # Freeze all layers first
        for param in self.model.parameters():
            param.requires_grad = False

        if option == 'last_layer':
            # Unfreeze only the last layer (original behavior)
            for param in self.model[-1][-1].parameters():
                param.requires_grad = True

        elif option == 'classifier':
            # Unfreeze the entire classifier
            for param in self.model[-1].parameters():
                param.requires_grad = True

        elif option == 'last_block':
            # Unfreeze the last convolutional block and classifier
            for param in self.model[0][-1].parameters():
                param.requires_grad = True
            for param in self.model[-1].parameters():
                param.requires_grad = True

        elif option == 'last_three_blocks':
            # Unfreeze the last three convolutional blocks and classifier
            for i in range(-3, 0):
                for param in self.model[0].parameters():
                    param.requires_grad = True
            for param in self.model[-1].parameters():
                param.requires_grad = True

        elif option == 'all':
            # Unfreeze all layers
            for param in self.model.parameters():
                param.requires_grad = True

    def forward(self, x):
        for block in self.model[:-1]:
            x = block(x)
        for layer in self.model[-1][:-1]:
            x = layer(x)

        return x, self.model[-1][-1](x)


class ConvNextLarge(nn.Module):
    def __init__(self, pretrained=None, unfreeze_option='last_layer',num_classes=1):
        super(ConvNextLarge, self).__init__()
    
        if pretrained:
            model = convnext_large(weights='ConvNeXt_Large_Weights.IMAGENET1K_V1')
        else:
            model = convnext_large(weights=None)
    
        if not pretrained:
            model.features[0][0] = nn.Conv2d(in_channels=1, out_channels=model.features[0][0].out_channels,
                                             kernel_size=4, stride=4)
        else:
            model = convert_first_layer_to_grayscale(model, model_type="ConvNext")

        model_blocks = get_layers_list(model)

        self.model = model_blocks

        if pretrained:
            self._freeze_layers(unfreeze_option)

    def _freeze_layers(self, option):
        # Freeze all layers first
        for param in self.model.parameters():
            param.requires_grad = False

        if option == 'last_layer':
            # Unfreeze only the last layer (original behavior)
            for param in self.model[-1][-1].parameters():
                param.requires_grad = True

        elif option == 'classifier':
            # Unfreeze the entire classifier
            for param in self.model[-1].parameters():
                param.requires_grad = True

        elif option == 'last_block':
            # Unfreeze the last convolutional block and classifier
            for param in self.model[0][-1].parameters():
                param.requires_grad = True
            for param in self.model[-1].parameters():
                param.requires_grad = True

        elif option == 'last_three_blocks':
            # Unfreeze the last three convolutional blocks and classifier
            for i in range(-3, 0):
                for param in self.model[0].parameters():
                    param.requires_grad = True
            for param in self.model[-1].parameters():
                param.requires_grad = True

        elif option == 'all':
            # Unfreeze all layers
            for param in self.model.parameters():
                param.requires_grad = True

    def forward(self, x):
        for block in self.model[:-1]:
            x = block(x)
        for layer in self.model[-1][:-1]:
            x = layer(x)

        return x, self.model[-1][-1](x)


if __name__ == "__main__":

    m2 = ConvNextBase(pretrained=True, unfreeze_option='all')

    test_img = torch.rand((6, 1, 512, 512))
    output1, output2 = m2(test_img)

    print(output1.shape)
    print(output2.shape)

    # trainable_params = sum(p.numel() for p in m2.parameters() if p.requires_grad)
    # print(f"Total number of trainable parameters: {trainable_params}")
    #
    #
    #
    # for name, paramater in m2.named_parameters():
    #     print(f"Layer: {name} | Size: {paramater.size()}")
