from training import run_training
from models import MLP,ProjMLP
from loaders import MNISTLoader,FashionMNISTLoader

import torch





if __name__ == '__main__':
    # model = ProjMLP(hidden_dim=16,n_hidden=1)
    model = MLP(hidden_dim=16,n_hidden=2)
    loaders = FashionMNISTLoader()
    train_loader = loaders.train_loader
    val_loader = loaders.val_loader   

    device = torch.device('cuda:0') 
                 
    run_training(model,train_loader,val_loader,device)
