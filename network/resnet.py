from torchvision.models import resnet34, resnet18

def ResNet18(num_classes=100):
    return resnet18(num_classes=100)
 
def ResNet34(num_classes=100):
    return resnet34(num_classes=100)