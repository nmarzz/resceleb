from datasets import CelebA
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import torchvision
import yaml

config = open('config.yaml', 'r')
parsed_config = yaml.load(config, Loader=yaml.FullLoader)

root = parsed_config['root_to_celeba']

def celeba_loader(batch_size: int, mode:str,distributed: bool = False, split_for_testing:str = None) -> tuple([DataLoader, DataLoader]):
    if split_for_testing is not None:
        if split_for_testing in ['train','val','test']:
            data_set = CelebA(root, mode=mode,split=split_for_testing,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])]
                ))
            return DataLoader(data_set, batch_size=batch_size, shuffle=False)
        else:
            raise ValueError('Specify a valid split for testing')

    # Get the train/val sets
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_set = CelebA(root, mode=mode,split='train',
            transform=train_transforms
            )

    val_set = CelebA(root, mode=mode,split='val',
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])]
            ))

    # For distributed training
    if distributed:
        sampler = DistributedSampler(train_set)
    else:
        sampler = None

    # Now make the loaders
    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=sampler, num_workers=1,
        pin_memory=True, shuffle=(not distributed))

    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader,val_loader



def cifar10_loader(batch_size: int, mode:str,distributed: bool = False) -> tuple([DataLoader, DataLoader]):
    # Get the train/val sets

    train_set = torchvision.datasets.CIFAR10('data', train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ]),download = True)

    val_set = torchvision.datasets.CIFAR10('data', train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ]),download = True)

    # For distributed training
    if distributed:
        sampler = DistributedSampler(train_set)
    else:
        sampler = None

    # Now make the loaders
    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=sampler, num_workers=1,
        pin_memory=True, shuffle=(not distributed))

    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader,val_loader
