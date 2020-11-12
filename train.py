import os
from argparse import ArgumentParser

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from dataset.dataset import SteelData
from model.model import ClsModel, Model
from util.loss import DiceLoss


def create_dataloader(swarg):
    train_csv = pd.read_csv(os.path.join(arg.root, 'train.csv'))
    train_dataset = SteelData(root=arg.root, mode='train', csv=train_csv)

    train_loader = DataLoader(train_dataset,
                              num_workers=arg.n_cpu,
                              shuffle=True,
                              drop_last=True,
                              batch_size=arg.batch_size)
    val_dataset = SteelData(root=arg.root, mode='val', csv=train_csv)
    val_loader = DataLoader(val_dataset,
                            num_workers=arg.n_cpu,
                            drop_last=True,
                            batch_size=arg.batch_size)
    return train_loader, val_loader


if __name__ == '__main__':
    parser = ArgumentParser('Steel Defect')
    parser.add_argument('--root', type=str, default='data/steel')
    parser.add_argument('--n_cpu', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--group', type=int, default=16, help="Unet groups")
    parser.add_argument('--lr', type=float, default=1e-3, help='defalut lr')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--mode', type=str, default='seg', help='seg or cls')
    parser.add_argument('--checkpoint', type=str, default='weights')
    parser.add_argument('--tpu',action='store_true')
    parser.add_argument('--decoder',
                        type=str,
                        default='unet',
                        help='efficient net  choose')
    parser.add_argument('--encoder',
                        type=str,
                        default='resnet34',
                        help='efficient net  choose')

    parser.add_argument('--radam', action='store_true')
    arg = parser.parse_args()
    print(arg)
    from util.loss import lovasz_softmax
    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()

    def seg_criterion(y_pred, y):
        bce_loss = bce(y_pred, y)
        dice_loss = dice(y_pred, y)
        return 0.6 * bce_loss + 0.4 * dice_loss
    if arg.mode == 'seg':
        checkpoint = ModelCheckpoint(
            dirpath=arg.checkpoint, verbose=True, mode='max', monitor='iou', filename=f'{arg.encoder}_{arg.decoder}')
        model = Model(criterion=seg_criterion,
                      encoder=arg.encoder,
                      decoder=arg.decoder)
    else:
        checkpoint = ModelCheckpoint(
            dirpath=arg.checkpoint, verbose=True, mode='max', monitor='acc', filename=f'{arg.encoder}')
        model = ClsModel(
            batch_size=arg.batch_size,
            criterion=nn.BCELoss(),
            encoder=arg.encoder
        )
    train_loader, val_loader = create_dataloader(arg)

    param = {
        'gpus':1
    }
    if arg.tpu:
        param = {
            'tpu_cores':8
        }
    trainer = pl.Trainer(
                         log_gpu_memory=True,
                         callbacks=[checkpoint],
                         benchmark=True,
                         accumulate_grad_batches=5,
                         max_epochs=arg.epochs,
                         auto_lr_find=True,
                         val_check_interval=0.5,**param)
    # log_gpu_memory=True, val_check_interval=0.5)
    if arg.mode == 'cls':
        trainer.tune(model, train_loader, val_loader)
    trainer.fit(model,
                train_dataloader=train_loader,
                val_dataloaders=val_loader)
