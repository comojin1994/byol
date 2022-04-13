'''Import libraries'''
import os, yaml

from datetime import datetime
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, random_split

from torchvision import transforms
from torchvision.datasets import CIFAR10
import torchvision.models as models

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from model.byol import ModelBase
from model.litmodel import LitModelLinear
from utils.setup_utils import get_device
from utils.training_utils import post_message


# Config setting
with open(f'configs/linear_config.yaml') as file:
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

args.lr = float(args.lr)
args.weight_decay = float(args.weight_decay)

### Set SEED ###
seed_everything(args.SEED)


def load_data():
    # Load data
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]) if args.cifar \
            else transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

    data = CIFAR10(root=args.DATA_PATH, train=True, transform=train_transform, download=True)
    train_data, val_data = random_split(data, [int(0.9 * len(data)), int(0.1 * len(data))])

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    args.num_classes = len(data.classes)
    
    return train_dataloader, val_dataloader


def load_pretrained_encoder():
    # Load model
    
    if args.cifar:
        model = models.__dict__[args.arch]
        model = ModelBase(model)
    else:
        model = models.__dict__[args.arch]()
    model.fc = nn.Linear(model.fc.in_features, args.num_classes, bias=True)

    # freeze all layers
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
                    
    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    # load pre-trained model
    checkpoint = torch.load(args.CKPT_PATH, map_location=f'cpu')
    state_dict = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        if k.startswith('model.online_encoder') and not k.startswith('model.online_encoder.fc'):
            state_dict[k[len('model.online_encoder.'):]] = state_dict[k]
        del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    
    return model


def main():
    
    train_dataloader, val_dataloader = load_data()
    
    ### Load model
    model = load_pretrained_encoder()
    model.cuda(int(args.GPU_NUM))
    model = LitModelLinear(model, args)
    
    
    ### Load logger and callbacks
    logger = TensorBoardLogger(args.LOG_PATH, 
                            name=f'{args.current_time}')


    ### Training
    trainer = Trainer(
        progress_bar_refresh_rate=20,
        max_epochs=args.EPOCHS,
        gpus=[int(args.GPU_NUM)],
        logger=logger
    )

    post_message("#manager", f'BYOL Linear classification Training Start, Time: {datetime.now().strftime("%Y/%m/%d-%H:%M:%S")}')
    
    trainer.fit(model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)
    
    post_message("#manager", f'BYOL Linear classification Training End, Time: {datetime.now().strftime("%Y/%m/%d-%H:%M:%S")}')


if __name__ == '__main__':
    import traceback
    
    try:
        main()
    except Exception as e:
        print(e)
        post_message("#manager", f'Error:\n\n{e}')
        print(traceback.format_exc())
        post_message("#manager", f'Traceback:\n\n{traceback.format_exc()}')
        
