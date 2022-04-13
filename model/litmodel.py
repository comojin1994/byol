import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import LightningModule

from sklearn.metrics import accuracy_score

from utils.optimizer import LARC


class LitModel(LightningModule):
    def __init__(self, model, memory_dataloader, test_dataloader, args):
        super().__init__()
        
        self.save_hyperparameters()
        self.model = model
        self.memory_dataloader = memory_dataloader
        self.test_dataloader = test_dataloader
        self.cur_lr = args.lr
        self.args = args
        
        
    def forward(self, x):
        return self.model(*x[0])
    
    
    def on_train_epoch_start(self):
        self.adjust_learning_rate()
        self.adjust_ema_decay_rate()        
    
    
    def training_step(self, batch, batch_idx):
        
        # self.adjust_learning_rate()
        # self.adjust_ema_decay_rate()
        
        p1, p2, z1, z2 = self(batch)
        return {'p': [p1, p2], 'z': [z1, z2]}
        # loss = self.criterion(p1, z2) + self.criterion(p2, z1)
        # loss = loss.mean()
        
        # self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # return {'loss': loss}
        
    
    def training_step_end(self, training_step_outputs):
        p1, p2 = training_step_outputs['p']
        z1, z2 = training_step_outputs['z']
        loss = self.criterion(p1, z2) + self.criterion(p2, z1)
        loss = loss.mean()
        
        self.model.update_target_weight()
        
        self.log('lr', self.cur_lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('decay_rate', self.model.decay_rate, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    
    def training_epoch_end(self, training_step_outputs):
        
        classes = self.args.num_classes
        total_top1, total_num, feature_bank = 0, 0, []
        
        with torch.no_grad():
            for data, target in self.memory_dataloader:
                feature = self.model.online_encoder(data.cuda(self.device.index, non_blocking=True))
                feature = F.normalize(feature, dim=1)
                feature_bank.append(feature)
            feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            feature_labels = torch.tensor(self.memory_dataloader.dataset.targets, device=feature_bank.device)

            for data, target in self.test_dataloader:
                feature = self.model.online_encoder(data.cuda(self.device.index, non_blocking=True))
                feature = F.normalize(feature, dim=1)

                pred_labels = self.knn_predict(feature, feature_bank, feature_labels, classes)
                
                total_num += data.size(0)
                total_top1 += accuracy_score(pred_labels[:, 0].cpu().numpy(), target) * data.size(0)
                
            acc = total_top1 / total_num * 100
            self.log(f'knn_acc', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            std = F.normalize(feature_bank, dim=0).std()
            self.log(f'std', std, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
    
    
    def configure_optimizers(self):
        # Optimizer
        parameters = list(filter(lambda p: p.requires_grad, self.parameters()))
        
        self.optimizer = torch.optim.SGD(parameters,
                                        self.args.lr,
                                        momentum=self.args.momentum,
                                        weight_decay=self.args.weight_decay)
        
        self.optimizer = LARC(optimizer=self.optimizer, trust_coefficient=0.001, clip=False)
        
        return {"optimizer": self.optimizer}
    
    
    # def on_before_zero_grad(self, _):
    #     self.model.update_target_weight()
    
    
    def adjust_learning_rate(self):
        self.cur_lr = self.args.lr * 0.5 * (1. + math.cos(math.pi * self.current_epoch / self.args.EPOCHS))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.cur_lr
            
            
    def adjust_ema_decay_rate(self):
        previous_decay_rate = self.args.EMA_decay_rate
        cur_decay_rate = 1. - 0.5 * (1. - previous_decay_rate) * (1. + math.cos(math.pi * self.current_epoch / self.args.EPOCHS))
        self.model.decay_rate = cur_decay_rate

    
    def criterion(self, x, y):
        normed_x = F.normalize(x, dim=1)
        normed_y = F.normalize(y, dim=1)
        
        loss = (((normed_x - normed_y)**2).sum(dim=1))
        return loss
    
    
    def knn_predict(self, feature, feature_bank, feature_labels, classes, knn_k=200, knn_t=0.1):
        sim_matrix = torch.mm(feature, feature_bank)
    
        sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)

        sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices) # check
        sim_weight = (sim_weight / knn_t).exp()

        one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)

        one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)

        pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

        pred_labels = pred_scores.argsort(dim=-1, descending=True)
        return pred_labels
    
    
class LitModelLinear(LightningModule):
    def __init__(self, model, args):
        super().__init__()
        
        self.save_hyperparameters()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.args = args
        
        
    def forward(self, x):
        return self.model(x)
    
    
    def training_step(self, batch, batch_idx):
        
        self.adjust_learning_rate()
        
        output = self(batch[0])
        loss = self.criterion(output, batch[1])
        
        acc1, acc5 = self.accuracy(output, batch[1], topk=(1, 5))
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_acc1', acc1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_acc5', acc5, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return {'loss': loss}
    
    
    def evaluate(self, batch, stage=None):
        
        output = self(batch[0])
        loss = self.criterion(output, batch[1])
        
        acc1, acc5 = self.accuracy(output, batch[1], topk=(1, 5))
        
        if stage:
            self.log(f'{stage}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f'{stage}_acc1', acc1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f'{stage}_acc5', acc5, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
    
    
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, stage='val')
    
    
    def test_step(self, batch, batch_idx):
        self.evaluate(batch, stage='test')
    
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return torch.argmax(self(batch[0]), dim=1)
    
    
    def configure_optimizers(self):
        parameters = list(filter(lambda p: p.requires_grad, self.parameters()))
        
        self.optimizer = torch.optim.SGD(parameters,
                                        self.args.lr,
                                        momentum=self.args.momentum,
                                        weight_decay=self.args.weight_decay)
        
        return {'optimizer': self.optimizer}
    
    
    def adjust_learning_rate(self):
        cur_lr = self.args.lr * 0.5 * (1. + math.cos(math.pi * self.current_epoch / self.args.EPOCHS))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr    
    
    
    def accuracy(self, output, target, topk=(1, 5)):
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
    