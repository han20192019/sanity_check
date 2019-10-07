#!/usr/bin/env python

import numpy as np
import torch
#import torch.nn.functional
import cfl
import sys

from deepinpy.utils import utils
import deepinpy.utils.complex as cp
import deepinpy.opt.conjgrad
from deepinpy.opt import opt
from deepinpy.utils import sim
from deepinpy.models.mcmri.mcmri import MultiChannelMRI
from deepinpy.models.resnet.resnet import ResNet5Block, ResNet

import torchvision.utils

import pytorch_lightning as pl

class ResNetRecon(pl.LightningModule):

    def __init__(self, args):
        super(ResNetRecon, self).__init__()
        self.loss_fun = torch.nn.MSELoss(reduction='sum')
        self.step = args.step
        self.solver = args.solver
        self.stdev = args.stdev
        self.num_data_sets = args.num_data_sets
        self.fully_sampled = args.fully_sampled
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.shuffle = args.shuffle

        self._build_data()

        if args.network == 'ResNet5Block':
            self.denoiser = ResNet5Block(num_filters=args.latent_channels, filter_size=7, batch_norm=args.batch_norm)
        elif args.network == 'ResNet':
            self.denoiser = ResNet(latent_channels=args.latent_channels, num_blocks=args.num_blocks, kernel_size=7, batch_norm=args.batch_norm)

    def _build_data(self):
        self.D = sim.Dataset(data_file="/home/jtamir/projects/deepinpy_git/data/dataset_train.h5", stdev=self.stdev, num_data_sets=self.num_data_sets, adjoint=False, id=0, clear_cache=False, cache_data=False, gen_masks=False, sure=False, scale_data=False, fully_sampled=self.fully_sampled, data_idx=None, inverse_crime=False)

    def _build_MCMRI(self, maps, masks):
        return MultiChannelMRI(maps, masks, 0.)

    def forward(self, x_adj):
        return self.denoiser(x_adj)

    def training_step(self, batch, batch_nb):
        idx, data = batch
        idx = utils.itemize(idx)
        imgs = data['imgs']
        maps = data['maps']
        masks = data['masks']
        inp = data['out']

        A = self._build_MCMRI(maps, masks)

        x_adj = A.adjoint(inp)
        x_hat = self.forward(x_adj)
        if 0 in idx:
            _idx = idx.index(0)
            cfl.writecfl('x_hat', utils.t2n(x_hat[_idx,...]))
            cfl.writecfl('x_gt', utils.t2n(imgs[_idx,...]))
            cfl.writecfl('masks', utils.t2n2(masks[_idx,...]))
            cfl.writecfl('maps', utils.t2n(maps[_idx,...]))
            cfl.writecfl('ksp', utils.t2n(inp[_idx,...]))
            myim = cp.zabs(torch.cat((x_hat[_idx,...], imgs[_idx,...]), dim=1))[None,None,...,0]
            grid = torchvision.utils.make_grid(myim, scale_each=True, normalize=True, nrow=1)
            self.logger.experiment.add_image('Train prediction', grid, 0)


        loss = self.loss_fun(x_hat, imgs)

        _loss = loss.clone().detach().requires_grad_(False)
        _epoch = self.current_epoch
        _nrmse = (opt.ip_batch(x_hat - imgs) / opt.ip_batch(imgs)).sqrt().mean().detach().requires_grad_(False)

        if self.logger:
            self.logger.log_metrics({
                'loss': _loss,
                'epoch': self.current_epoch,
                'nrmse': _nrmse, 
                })
        return {
                'loss': loss
                }

    def configure_optimizers(self):
        if 'adam' in self.solver:
            return [torch.optim.Adam(self.parameters(), lr=self.step)]
        elif 'sgd' in self.solver:
            return [torch.optim.SGD(self.parameters(), lr=self.step)]

    @pl.data_loader
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.D, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, drop_last=True)
