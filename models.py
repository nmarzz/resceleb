from torch import nn
import torch.nn.functional as F
import numpy as np
import torch


class MLP(nn.Module):
    def __init__(self,hidden_dim : int = 32,n_hidden:int=1,dropout:bool = False) -> None:
        super(MLP,self).__init__()                
        self.fc1 = nn.Linear(28*28, hidden_dim)   
        self.linears = nn.ModuleList([nn.Linear(hidden_dim,hidden_dim) for _ in range(n_hidden)])                     
        self.fc2 = nn.Linear(hidden_dim,hidden_dim,bias=False)        
        self.fc3 = nn.Linear(hidden_dim,10)


    def forward(self,x):        
        x = x.view(-1,28*28)                
        x = F.relu(self.fc1(x))    
        for layer in self.linears:            
            x = F.relu(layer(x))                                    
        x = self.fc3(x)
        return x
        

class ProjMLP(nn.Module):
    def __init__(self,hidden_dim:int = 32,n_hidden:int = 1) -> None:
        super(ProjMLP,self).__init__()
        self.fc1 = nn.Linear(28*28, hidden_dim)                
        self.random_linears = nn.ModuleList([RandomLinear(hidden_dim,hidden_dim) for _ in range(n_hidden)])                
        # self.random_linears = RandomLinear(hidden_dim,hidden_dim)        
        self.fc3 = nn.Linear(hidden_dim,10)

    def forward(self,x):
        x = x.view(-1,28*28)                
        x = F.relu(self.fc1(x))       
        # x = F.relu(self.random_linears(x))                         
        for layer in self.random_linears:            
            x = F.relu(layer(x))        
        x = self.fc3(x)
        return x
        

        
class RandomLinear(nn.Module):
    def __init__(self,in_dim,out_dim) -> None:
        super(RandomLinear,self).__init__()    
        if in_dim != out_dim:
            raise NotImplementedError('Random Linear layers must be square right now')
        
        dim = in_dim
        G1 = torch.tensor(np.random.choice([-1,1],size= (dim,dim)),requires_grad = False,dtype=torch.float32)
        G2 = torch.tensor(np.random.choice([-1,1],size= (dim,dim)),requires_grad = False,dtype=torch.float32)
        self.register_buffer('G1',G1)
        self.register_buffer('G2',G2)

        self.Lambda = nn.parameter.Parameter(torch.randn(dim))

    def forward(self,x):                
        x = x @ (self.G2 @ torch.diag(self.Lambda) @ self.G1)
        return x
