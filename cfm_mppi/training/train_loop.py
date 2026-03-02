# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import gc
import logging
import math
from typing import Iterable

import torch
from cfm_mppi.flow_matching.path import CondOTProbPath
from models.ema import EMA
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.aggregation import MeanMetric
from training.grad_scaler import NativeScalerWithGradNormCount
from einops import rearrange, repeat

MASK_TOKEN = 256
PRINT_FREQUENCY = 50
TIME_STEP = 0.1


def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    time = 1 / (1 + sigma)
    time = torch.clip(time, min=0.0001, max=1.0)
    return time


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    lr_schedule: torch.torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    loss_scaler: NativeScalerWithGradNormCount,
    args: argparse.Namespace,
):
    gc.collect()
    model.train(True)
    batch_loss = MeanMetric().to(device, non_blocking=True)
    epoch_loss = MeanMetric().to(device, non_blocking=True)

    accum_iter = args.accum_iter

    path = CondOTProbPath()

    for data_iter_step, samples in enumerate(data_loader):
        if data_iter_step % accum_iter == 0:
            optimizer.zero_grad()
            batch_loss.reset()
            if data_iter_step > 0 and args.test_run:
                break


        samples = samples.to(device, non_blocking=True)
        samples = samples/args.scale
        pos = samples[:,:2,:]
        control = samples[:,2:4,:]

        # Scaling to [-1, 1] from [0, 1]
        # samples = samples * 2.0 - 1.0
        noise = torch.randn_like(control).to(device)
        if args.skewed_timesteps:
            t = skewed_timestep_sample(control.shape[0], device=device)
        else:
            t = torch.torch.rand(control.shape[0]).to(device)


        path_sample = path.sample(t=t, x_0=noise, x_1=control)

        x_t = path_sample.x_t
        u_t = path_sample.dx_t

        with torch.cuda.amp.autocast():
            u_t_pred = model(x_t, t, start=pos[:,:,0], goal=pos[:,:,-1])

            loss = torch.nn.functional.mse_loss(u_t_pred, u_t)

        loss_value = loss.item()
        batch_loss.update(loss)
        epoch_loss.update(loss)


        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss /= accum_iter

        # Loss scaler applies the optimizer when update_grad is set to true.
        # Otherwise just updates the internal gradient scales
        apply_update = (data_iter_step + 1) % accum_iter == 0
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=apply_update,
        )
        if apply_update and isinstance(model, EMA):
            model.update_ema()
        elif (
            apply_update
            and isinstance(model, DistributedDataParallel)
            and isinstance(model.module, EMA)
        ):
            model.module.update_ema()

        lr = optimizer.param_groups[0]["lr"]


    lr_schedule.step()
    return {"loss": float(epoch_loss.compute().detach().cpu())}
