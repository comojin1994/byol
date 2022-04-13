
'''Import libraries'''
import os, yaml

from datetime import datetime
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import CIFAR10
import torchvision.models as models

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from model.byol import BYOL
from model.litmodel import LitModel
from utils.setup_utils import get_device
from utils.data_utils import TwoCropsTransform
from utils.training_utils import get_callbacks, post_message


# Config setting
with open(f'configs/byol_config.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
    args = EasyDict(config)
    
args.current_time = datetime.now().strftime('%Y%m%d')

### Set Device ###
if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_NUM
    
args['device'] = get_device(args.GPU_NUM)
cudnn.benchmark = True
cudnn.fastest = True
cudnn.deterministic = True

args.lr = float(args.lr) * args.batch_size / 256
args.weight_decay = float(args.weight_decay)

### Set SEED ###
seed_everything(args.SEED)


def load_data():
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]) if args.cifar \
        else transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = [
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize
        ]

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    train_data = CIFAR10(root=args.DATA_PATH, train=True, transform=TwoCropsTransform(transforms.Compose(train_transform)), download=True)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=args.num_workers)

    memory_data = CIFAR10(root=args.DATA_PATH, train=True, transform=test_transform, download=True)
    memory_dataloader = DataLoader(memory_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    test_data = CIFAR10(root=args.DATA_PATH, train=False, transform=test_transform, download=True)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    args.num_classes = len(train_data.classes)
    
    return train_dataloader, memory_dataloader, test_dataloader


def main():
    train_dataloader, memory_dataloader, test_dataloader = load_data()
    
    ### Load model
    byol = BYOL(models.__dict__[args.arch], projection_dim=args.projection_dim, hidden_pro_dim=args.hidden_projection_dim, hidden_pre_dim=args.hidden_prediction_dim, decay_rate=args.EMA_decay_rate, cifar=True)
    model = LitModel(byol, memory_dataloader, test_dataloader, args)
    
    ### Load logger and callbacks
    logger = TensorBoardLogger(args.LOG_PATH, name=f'{args.current_time}')
    callbacks = get_callbacks(args)    


    ### Training
    trainer = Trainer(
        strategy='dp',
        progress_bar_refresh_rate=20,
        max_epochs=args.EPOCHS,
        gpus=list(map(int, args.GPU_NUM.split(','))),
        callbacks=callbacks,
        logger=logger
    )

    post_message("#manager", f'BYOL Training Start, Time: {datetime.now().strftime("%Y/%m/%d-%H:%M:%S")}')
    
    trainer.fit(model, train_dataloaders=train_dataloader)
    
    post_message("#manager", f'BYOL Training End, Time: {datetime.now().strftime("%Y/%m/%d-%H:%M:%S")}')


if __name__ == '__main__':
    import traceback
    
    try:
        main()
    except Exception as e:
        print(e)
        post_message("#manager", f'Error:\n\n{e}')
        print(traceback.format_exc())
        post_message("#manager", f'Traceback:\n\n{traceback.format_exc()}')     

