from models.resnet18 import ResNet18v1
from models.resnet50 import ResNet50v1
from models.resnet101 import ResNet101v1
from models.resnet152 import ResNet152v1
from models.convnext import *


def get_model(model_name):
    if model_name == "resnet50":
        print(f"using resnet50")
        model = ResNet50v1(unfreeze_option="last_three_blocks")
    elif model_name == "resnet18":
        model = ResNet18v1()
        print(f"using resnet18")
    elif model_name == "resnet101":
        model = ResNet101v1(unfreeze_option="last_three_blocks")
        print(f"using resnet101")
    elif model_name == "resnet152":
        model = ResNet152v1(unfreeze_option="last_three_blocks")
        print(f"using resnet152")
    elif model_name == "ConvNextTiny":
        model = ConvNextTiny(unfreeze_option="last_three_blocks")
        print(f"using ConvNextTiny")
    elif model_name == "ConvNextSmall":
        model = ConvNextSmall(unfreeze_option="last_three_blocks")
        print(f"using ConvNextSmall")
    elif model_name == "ConvNextBase":
        model = ConvNextBase(unfreeze_option="last_three_blocks")
        print(f"using ConvNextBase")
    else:
        model = ConvNextLarge(unfreeze_option="last_three_blocks")
        print(f"using ConvNextLarge")

    return model