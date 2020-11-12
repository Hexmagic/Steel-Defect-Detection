from typing import IO, Optional

import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.utils.functional import _threshold
from segmentation_models_pytorch.utils.metrics import Accuracy, IoU
from torch.optim import AdamW, lr_scheduler

from .unet_plus import UnetPP


def cal_dice(pred, target):
    with torch.no_grad():
        pred = torch.sigmoid(pred[:, 1:])
        target = target[:, 1:]
        batch = len(pred)
        pred = _threshold(pred, 0.5)
        p = pred.view(batch, -1)
        t = target.view(batch, -1)
        intersection = (p * t).sum(-1)
        union = (p + t).sum(-1)
        dice = ((2 * intersection) / (union+1e-5)).mean().item()
        return dice


class ClsModel(pl.LightningModule):
    def __init__(self, criterion, batch_size
    , threshold=0.5, num_class=5, encoder='resnet34', lr=1e-3):
        super(ClsModel, self).__init__()
        self.criterion = criterion
        self.lr = lr
        self.threshold = threshold
        self.num_class = num_class
        self.encoder = get_encoder(encoder)
        self.classify = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 64),
            nn.Dropout(),
            nn.Linear(64, self.num_class),
            nn.Sigmoid()
        )
        self.batch_size = batch_size
        self.accs = []
        self.best_acc = 0
        self.acc_cal = Accuracy()

    def forward(self, x):
        return self.classify(self.encoder(x)[-1])

    def training_step(self, batch, batch_idx):
        x, _, y = batch
        y_pred = self.classify(self.encoder(x)[-1])
        loss = self.criterion(y_pred, y)
        self.log("train_cls_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, y = batch
        y_pred = self.classify(self.encoder(x)[-1])
        loss = self.criterion(y_pred, y)
        y_pred = (y_pred > self.threshold).type(torch.float)
        self.accs.append(self.acc_cal(y_pred, y).item())
        self.log("val_cls_loss", loss)
        return loss

    def validation_epoch_end(self, *args, **kwargs):
        self.log("acc", np.mean(self.accs))
        acc = np.mean(self.accs)
        if acc > self.best_acc:
            self.best_acc = acc
            print(f'better acc: {self.best_acc}')
        self.accs = []

    def configure_optimizers(self):
        optimizer = AdamW(self.encoder.parameters(), lr=self.lr)
        return optimizer


class Model(pl.LightningModule):
    def __init__(self,
                 criterion=None,
                 num_class=5,
                 decoder='unet',
                 threshold=0.5,
                 lr=1e-3,
                 encoder='resnet34') -> None:
        super(Model, self).__init__()
        self.best_iou = 0
        self.best_dice = 0
        self.lr = lr
        self.dices = []
        self.ious = []
        self.num_class = num_class
        decoder_map = {'unet': smp.Unet, 'fpn': smp.FPN,
                       'psp': smp.PSPNet, 'unetpp': UnetPP}
        self.criterion = criterion
        self.iou_cal = IoU(ignore_channels=[0])
        self.threshold = threshold
        self.decoder = decoder_map[decoder](encoder, encoder_weights='imagenet', classes=num_class,
                                            activation=None)

    def forward(self, x):
        return self.decoder(x)

    def training_step(self, batch, batch_idx):
        x, mask, _ = batch
        y_pred = self.decoder(x)
        loss = self.criterion(y_pred, mask)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.decoder.parameters(), lr=self.lr)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=2, verbose=True, factor=0.5, eps=1e-6, min_lr=1e-5)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'train_loss'}

    def validation_step(self, batch, batch_idx):
        x, mask, _ = batch
        y_pred = self.decoder(x)
        loss = self.criterion(y_pred, mask)
        iou = self.iou_cal(y_pred, mask)
        self.dices.append(cal_dice(y_pred, mask))
        self.ious.append(iou.item())
        self.log("val_loss", loss)
        return loss

    def validation_epoch_end(self, *args, **kwargs):
        self.log("iou", np.mean(self.ious))
        self.log("dice", np.mean(self.dices))
        mean_iou = np.mean(self.ious)
        mean_dice = np.mean(self.dices)
        if mean_iou > self.best_iou:
            self.best_iou = mean_iou
            print(f'better iou: {self.best_iou}')
        if mean_dice > self.best_dice:
            self.best_dice = mean_dice
            print(f'better dice: {self.best_dice}')
        self.log("iou", mean_iou, prog_bar=True)
        self.ious = []
        self.dices = []
