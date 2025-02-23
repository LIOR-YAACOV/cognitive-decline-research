from models.resnet18 import ResNet18v1
from models.resnet50 import ResNet50v1
from models.resnet101 import ResNet101v1
from models.resnet152 import ResNet152v1
from models.convnext import *


def get_model(configure):
    unfreeze_option = configure["unfreeze"]
    if not configure["pretrained"]:
        if configure["model"] == "resnet50":
            print(f"using resnet50")
            model = ResNet50v1(unfreeze_option=unfreeze_option)
        elif configure["model"] == "resnet18":
            model = ResNet18v1()
            print(f"using resnet18")
        elif configure["model"] == "resnet101":
            model = ResNet101v1(unfreeze_option=unfreeze_option)
            print(f"using resnet101")
        elif configure["model"] == "resnet152":
            model = ResNet152v1(unfreeze_option=unfreeze_option)
            print(f"using resnet152")
        elif configure["model"] == "ConvNextTiny":
            model = ConvNextTiny(unfreeze_option=unfreeze_option)
            print(f"using ConvNextTiny")
        elif configure["model"] == "ConvNextSmall":
            model = ConvNextSmall(unfreeze_option=unfreeze_option)
            print(f"using ConvNextSmall")
        elif configure["model"] == "ConvNextBase":
            model = ConvNextBase(unfreeze_option=unfreeze_option)
            print(f"using ConvNextBase")
        else:
            model = ConvNextLarge(unfreeze_option=unfreeze_option)
            print(f"using ConvNextLarge")
    else:
        if configure["model"] == "resnet50":
            print(f"using imagenet pretrained resnet50")
            model = ResNet50v1(imagenet_pretrained=True, unfreeze_option=unfreeze_option)
        elif configure["model"] == "resnet18":
            model = ResNet18v1(imagenet_pretrained=True)
            print(f"using imagenet pretrained resnet18")
        elif configure["model"] == "resnet101":
            model = ResNet101v1(imagenet_pretrained=True, unfreeze_option=unfreeze_option)
            print(f"using imagenet pretrained resnet101")
        elif configure["model"] == "resnet152":
            model = ResNet152v1(imagenet_pretrained=True, unfreeze_option=unfreeze_option)
            print(f"using imagenet pretrained resnet152")
        elif configure["model"] == "ConvNextTiny":
            model = ConvNextTiny(pretrained=True, unfreeze_option=unfreeze_option)
            print(f"using pretrained ConvNextTiny")
        elif configure["model"] == "ConvNextSmall":
            model = ConvNextSmall(pretrained=True, unfreeze_option=unfreeze_option)
            print(f"using pretrained ConvNextSmall")
        elif configure["model"] == "ConvNextBase":
            model = ConvNextBase(pretrained=True, unfreeze_option=unfreeze_option)
            print(f"using pretrained ConvNextBase")
        else:
            model = ConvNextLarge(pretrained=True, unfreeze_option=unfreeze_option)
            print(f"using pretrained ConvNextLarge")

    return model

