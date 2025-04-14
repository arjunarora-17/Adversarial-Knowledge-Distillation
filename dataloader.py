from torchvision import datasets, transforms
import torch

def get_dataloader(args):
    test_loader = torch.utils.data.DataLoader( 
        datasets.CIFAR100(args.data_root, train=False, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])),
        batch_size=args.batch_size, shuffle=True, num_workers=2)
    return test_loader