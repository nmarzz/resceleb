from training import ResNet50_Trainer
from torchvision.models import resnet50
from loaders import celeba_loader,cifar10_loader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp

import argparse
import torch
import numpy as np
from logger import Logger
import yaml

config = open('config.yaml', 'r')
parsed_config = yaml.load(config, Loader=yaml.FullLoader)

def get_args(parser):
    """Collect command line arguments."""
    parser.add_argument('--mode', type=str, choices=['hair_color','hair_style'] ,metavar='M')
    parser.add_argument('--device', type=str, nargs='+', default=['cpu'],
                        help='Name of CUDA device(s) being used (if any). Otherwise will use CPU. \
                            Can also specify multiple devices (separated by spaces) for multiprocessing.')
    parser.add_argument('--optimizer', type=str, choices=['sgd','adam'] ,metavar='O')    
    parser.add_argument('--lr', type=float, default = 1e-3)                                                      
    parser.add_argument('--epochs', type=int, default = 10)                                                      
    parser.add_argument('--batch-size', type=int, default = 256)    
    parser.add_argument('--seed', type=int, default = 1331)                                                      

    args = parser.parse_args()

    return args


def main_worker(idx:int,num_gpus:int,distributed:bool,args:argparse.Namespace):        
    device = torch.device(args.device[idx])   
    num_classes = len(parsed_config['hair_color_vals']) if args.mode == 'hair_color' else len(parsed_config['hair_style_vals']) 
    
    logger = Logger(args, save=(idx == 0))

    if distributed:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:29501',
            world_size=num_gpus, rank=idx)
        

    #Get the data
    batch_size = int(args.batch_size / num_gpus)
    train_loader, val_loader = celeba_loader(batch_size=batch_size,mode = args.mode,distributed = distributed)        
    
    # To test on cifar as it is easier/faster
    # train_loader, val_loader = cifar10_loader(batch_size=batch_size,mode = args.mode,distributed = distributed)        
    # num_classes = 10
                
    model = resnet50(num_classes = num_classes)
    model.to(device)


    if distributed:
        model = DistributedDataParallel(model, device_ids=[device])

    trainer = ResNet50_Trainer(model = model,train_loader = train_loader,val_loader = val_loader,device = device,logger=logger,idx = idx,args=args)    
    trainer.train()

    
    

def main():
    """Load arguments, the dataset, and initiate the training loop."""
    parser = argparse.ArgumentParser(description='Training on CelebA')
    args = get_args(parser)

    if args.mode is None:
        raise ValueError('Must specify a training mode')

    #Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    num_gpus = len(args.device)
    
    #If we are doing distributed computation over multiple GPUs
    if num_gpus > 1:
        mp.spawn(main_worker, nprocs=num_gpus, args=(num_gpus, True, args))
    else:
        main_worker(0, 1, False, args)

if __name__ == '__main__':
    main()
