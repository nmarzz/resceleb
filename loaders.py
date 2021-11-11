import torch
from torchvision import datasets, transforms



class MNISTLoader():
    def __init__(self,batch_size:int = 128) -> None:        
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, 
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]),download = True), batch_size=batch_size, shuffle=True)


        self.val_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ]),download = True), batch_size=1000, shuffle=True)


class FashionMNISTLoader():
    def __init__(self,batch_size:int = 128) -> None:        
        self.train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('data', train=True, 
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]),download = True), batch_size=batch_size, shuffle=True)


        self.val_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ]),download = True), batch_size=1000, shuffle=True)                        