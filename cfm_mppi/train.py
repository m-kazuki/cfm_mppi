import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from cfm_mppi.models.transformer import TransformerModel
from train_arg_parser import get_args_parser

from training import distributed_mode
from training.grad_scaler import NativeScalerWithGradNormCount as NativeScaler
from training.load_and_save import load_model, save_model
from training.train_loop import train_one_epoch
import pickle
import wandb
import random


config = wandb.config


class LightDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)

        self.n_data = self.data.shape[0]
    
    def __len__(self):
        return self.n_data
    
    def __getitem__(self, idx):
        return self.data[idx]
    


def random_collate_fn(batch):
    """
    batch
    """
    L = random.randint(10, 80)

    batch_cut= torch.stack(batch, dim=0)
    batch_cut = batch_cut[:,:,:L]

    return batch_cut

def main(args):
    model_name = args.model_arch

    output_dir = args.output_dir + '/' + model_name
    Path(output_dir).mkdir(parents=True, exist_ok=True)
        
    wandb.init(
        project="flow_matching_transformer",
        name=model_name,
        config={
            "learning_rate": args.lr,
            'decay_lr': args.decay_lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "architecture": args.model_arch,
        }
    )
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    distributed_mode.init_distributed_mode(args)

    if distributed_mode.is_main_process():
        args_filepath = Path(output_dir) / "args.json"
        with open(args_filepath, "w") as f:
            json.dump(vars(args), f)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + distributed_mode.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # make dataset
    dataset_train = LightDataset("dataset/train80_ego.pt")

    num_tasks = distributed_mode.get_world_size()
    global_rank = distributed_mode.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )


    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        collate_fn=random_collate_fn,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the model
    model = TransformerModel()

    model.to(device)

    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(
        model_without_ddp.parameters(), lr=args.lr, betas=args.optimizer_betas
    )
    if args.decay_lr:
        lr_schedule = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=args.epochs,
            start_factor=1.0,
            end_factor=1e-8 / args.lr,
        )
    else:
        lr_schedule = torch.optim.lr_scheduler.ConstantLR(
            optimizer, total_iters=args.epochs, factor=1.0
        )


    loss_scaler = NativeScaler()

    load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        lr_schedule=lr_schedule,
    )

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if not args.eval_only:
            train_stats = train_one_epoch(
                model=model,
                data_loader=data_loader_train,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                device=device,
                epoch=epoch,
                loss_scaler=loss_scaler,
                args=args,
            )
            log_stats = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_stats.items()},
            }
            wandb.log({
                'epoch': epoch,
                'train_loss': train_stats['loss'],
            })
        else:
            log_stats = {
                "epoch": epoch,
            }

        if output_dir and (
            (args.eval_frequency > 0 and (epoch + 1) % args.eval_frequency == 0)
            or args.eval_only
            or args.test_run
        ):
            if not args.eval_only:
                save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    lr_schedule=lr_schedule,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                    output_dir=output_dir,
                )
            if args.distributed:
                data_loader_train.sampler.set_epoch(0)

        if output_dir and distributed_mode.is_main_process():
            with open(
                os.path.join(output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")


        if args.test_run or args.eval_only:
            break


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
