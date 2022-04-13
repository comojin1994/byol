from collections import OrderedDict

import torch
import torch.nn as nn


class ModelBase(nn.Sequential):
    """
    Common CIFAR ResNet recipe.
    Comparing with ImageNet ResNet recipe, it:
    (i) replaces conv1 with kernel=3, str=1
    (ii) removes pool1
    """
    def __init__(self, base_encoder, feature_dim=128, bn_splits=8):
        resnet_arch = base_encoder
        net = resnet_arch(num_classes=feature_dim)

        self.net = OrderedDict()
        for name, module in net.named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if isinstance(module, nn.MaxPool2d):
                continue
            if isinstance(module, nn.Linear):
                self.net['flatten'] = nn.Flatten(1)
            self.net[name] = module
        
        super(ModelBase, self).__init__(self.net)


class BYOL(nn.Module):
    def __init__(self, base_encoder, projection_dim=256, hidden_pro_dim=4096, hidden_pre_dim=4096, decay_rate=0.99, cifar=True):
        super(BYOL, self).__init__()
        
        self.decay_rate = decay_rate
        
        self.online_encoder = ModelBase(base_encoder, feature_dim=projection_dim) if cifar else base_encoder(num_classes=projection_dim, zero_init_residual=True)
        self.target_encoder = ModelBase(base_encoder, feature_dim=projection_dim) if cifar else base_encoder(num_classes=projection_dim, zero_init_residual=True)
        prev_dim = self.online_encoder.fc.weight.shape[1]
        
        # projector
        self.online_encoder.fc = nn.Sequential(nn.Linear(prev_dim, hidden_pro_dim, bias=True),
                                        nn.BatchNorm1d(hidden_pro_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_pro_dim, projection_dim, bias=False))
        self.target_encoder.fc = nn.Sequential(nn.Linear(prev_dim, hidden_pro_dim, bias=True),
                                        nn.BatchNorm1d(hidden_pro_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_pro_dim, projection_dim, bias=False))
        
        
        for param_online, param_target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_target.data.copy_(param_online.data)
            param_target.requires_grad = False
        
        
        # predictor
        self.predictor = nn.Sequential(nn.Linear(projection_dim, hidden_pre_dim, bias=True),
                                    nn.BatchNorm1d(hidden_pre_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_pre_dim, projection_dim, bias=False))

    
    def forward(self, x1, x2):
        
        online_z1 = self.online_encoder(x1)
        online_z2 = self.online_encoder(x2)
        
        # with torch.no_grad():
        target_z1 = self.target_encoder(x1)
        target_z2 = self.target_encoder(x2)
        
        p1 = self.predictor(online_z1)
        p2 = self.predictor(online_z2)
        
        return p1, p2, target_z1.detach(), target_z2.detach()
    
    
    @torch.no_grad()
    def update_target_weight(self):
        for param_online, param_target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_target = param_target.data * self.decay_rate + param_online.data * (1 - self.decay_rate)
    
    