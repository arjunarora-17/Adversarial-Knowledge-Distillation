from torchvision.models import resnet34, resnet18, efficientnet_b3

def ResNet18(num_classes=100):
    return resnet18(weights=None,num_classes=num_classes)

def ResNet34(num_classes=100):
    return resnet34(weights=None,num_classes=num_classes)

def EfficientNetB3(num_classes=100):
    return efficientnet_b3(weights=None, num_classes=num_classes)
