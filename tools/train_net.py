#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

import math
import numpy as np
import pprint
import torch
from torch.nn import functional as F
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
from pytorchvideo.layers.distributed import get_local_rank
import copy 

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.datasets.mixup import MixUp
from slowfast.models import build_model
from slowfast.models.contrastive import (
    contrastive_forward,
    contrastive_parameter_surgery,
)
from slowfast.utils.meters import AVAMeter, EpochTimer, TrainMeter, ValMeter
from slowfast.utils.multigrid import MultigridSchedule
from slowfast.utils.env import pathmgr
import os
import random
logger = logging.get_logger(__name__)



def compute_fisher_matrix_diag(
    train_loader,
    model,
    optimizer,
    scaler,
    train_meter,
    cur_epoch,
    cfg,
    writer=None,
):
    """
    Calculate the fisher matrix diag.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    fisher = {}
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)
    
    if cfg.MODEL.FROZEN_BN:
        misc.frozen_bn_stats(model)
    # Explicitly declare reduction to mean.
    loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="none")
    n_samples = 0
     
    for cur_iter, (inputs, labels, index, time, meta) in enumerate(
        train_loader
    ):
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    if isinstance(inputs[i], (list,)):
                        for j in range(len(inputs[i])):
                            inputs[i][j] = inputs[i][j].cuda(non_blocking=True)
                    else:
                        inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            if not isinstance(labels, list):
                labels = labels.cuda(non_blocking=True)
                index = index.cuda(non_blocking=True)
                time = time.cuda(non_blocking=True)
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

        batch_size = (
            inputs[0][0].size(0)
            if isinstance(inputs[0], list)
            else inputs[0].size(0)
        )
        # Update the learning rate.
        epoch_exact = cur_epoch + float(cur_iter) / data_size
        lr = optim.get_epoch_lr(epoch_exact, cfg)
        optim.set_lr(optimizer, lr)
         
        train_meter.data_toc()
        with torch.cuda.amp.autocast(enabled=False):

            # Explicitly declare reduction to mean.
            perform_backward = True
            optimizer.zero_grad()
            preds = model(inputs)
            
            # Compute the loss.
            loss = loss_fun(preds, labels)
            mask = (torch.max(preds, -1)[1] == labels).float()
            useful_sample_num = int(torch.sum(mask).item())
            loss = (loss * mask).mean()
            n_samples += useful_sample_num

        loss_extra = None
        if isinstance(loss, (list, tuple)):
            loss, loss_extra = loss

        # check Nan Loss.
        misc.check_nan_losses(loss)
        loss.backward()
        # if perform_backward:
        #     scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        # scaler.unscale_(optimizer)
        # gradients norm calculate
        grad_norm = optim.get_grad_norm_(model.parameters())
        
        # accumulate fisher matrix
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            
            if name not in fisher:
                fisher[name] = p.grad.pow(2) * useful_sample_num
            else:
                fisher[name] += p.grad.pow(2) * useful_sample_num

        # Update the parameters. (defaults to True)
        model, update_param = contrastive_parameter_surgery(
            model, cfg, epoch_exact, cur_iter
        )
        """ 
        if update_param:
            scaler.step(optimizer)
        """
        # scaler.update()

        top1_err, top5_err = None, None
        
        # Compute the errors.
        num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
        top1_err, top5_err = [
            (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
        ]
        # Gather all the predictions across all the devices.
        if cfg.NUM_GPUS > 1:
            loss, grad_norm, top1_err, top5_err = du.all_reduce(
                [loss.detach(), grad_norm, top1_err, top5_err]
            )
        
        # Copy the stats from GPU to CPU (sync point).
        loss, grad_norm, top1_err, top5_err = (
            loss.item(),
            grad_norm.item(),
            top1_err.item(),
            top5_err.item(),
        )

        # Update and log stats.
        train_meter.update_stats(
            top1_err,
            top5_err,
            loss,
            lr,
            grad_norm,
            batch_size
            * max(
                cfg.NUM_GPUS, 1
            ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            loss_extra,
        )
        # write to tensorboard format if available.
        if writer is not None:
            writer.add_scalars(
                {
                    "Train/loss": loss,
                    "Train/lr": lr,
                    "Train/Top1_err": top1_err,
                    "Train/Top5_err": top5_err,
                },
                global_step=data_size * cur_epoch + cur_iter,
            )
        train_meter.iter_toc()  # do measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        torch.cuda.synchronize()
        train_meter.iter_tic()
    
    
    for i, key in enumerate(fisher):
        if i == 0:
            device = fisher[key].device
        fisher[key] /= n_samples
    
    test = du.all_gather_unaligned(fisher)
    
    gather_fisher = {}
    for key in fisher:
        gather_fisher[key] = []
    
    for key in gather_fisher:
        for partial_fisher in test:
            gather_fisher[key].append(partial_fisher[key].to(device))

    for key in gather_fisher:
        gather_fisher[key] = torch.stack(gather_fisher[key]).mean(0)
     
    if cfg.TRAIN.EWC_IDENTITY_FISHER:
        for key in gather_fisher:
            gather_fisher[key] = gather_fisher[key] * 0. + 1.

    torch.save(gather_fisher, os.path.join(cfg.OUTPUT_DIR, 'fisher_%d.pth'%get_local_rank()))
    del inputs
    # in case of fragmented memory
    torch.cuda.empty_cache()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()
    
    return gather_fisher

def train_epoch(
    train_loader,
    model,
    optimizer,
    scaler,
    train_meter,
    cur_epoch,
    cfg,
    writer=None,
    fisher_map=None,
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    if cfg.MIXUP.ENABLE:
        mixup_fn = MixUp(
            mixup_alpha=cfg.MIXUP.ALPHA,
            cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA,
            mix_prob=cfg.MIXUP.PROB,
            switch_prob=cfg.MIXUP.SWITCH_PROB,
            label_smoothing=cfg.MIXUP.LABEL_SMOOTH_VALUE,
            num_classes=cfg.MODEL.NUM_CLASSES,
        )

    if cfg.MODEL.FROZEN_BN:
        misc.frozen_bn_stats(model)
    # Explicitly declare reduction to mean.
    loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
    
    # save the raw clip weight
    if cfg.TRAIN.LINEAR_CONNECT_CLIMB:
        assert cfg.TRAIN.CLIP_ORI_PATH != None
        prev_weights = torch.jit.load(cfg.TRAIN.CLIP_ORI_PATH, map_location=model.device).state_dict()
        _ = [prev_weights.pop(i) for i in ['input_resolution', 'context_length', 'vocab_size']]
        # float32
        for key in prev_weights:
            prev_weights[key] = prev_weights[key].float()

    if cfg.MODEL.KEEP_RAW_MODEL:
        raw_clip_params = {}
        for n, p in model.named_parameters():
            if 'raw_model' in n:
                p.requires_grad = False
                raw_clip_params[n] = p
    
    # extract raw clip params

    for cur_iter, (inputs, labels, index, time, meta) in enumerate(
        train_loader
    ):
        
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    if isinstance(inputs[i], (list,)):
                        for j in range(len(inputs[i])):
                            inputs[i][j] = inputs[i][j].cuda(non_blocking=True)
                    else:
                        inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            if not isinstance(labels, list):
                labels = labels.cuda(non_blocking=True)
                index = index.cuda(non_blocking=True)
                time = time.cuda(non_blocking=True)
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

        batch_size = (
            inputs[0][0].size(0)
            if isinstance(inputs[0], list)
            else inputs[0].size(0)
        )
        # Update the learning rate.
        epoch_exact = cur_epoch + float(cur_iter) / data_size
        lr = optim.get_epoch_lr(epoch_exact, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()
        if cfg.MIXUP.ENABLE:
            samples, labels = mixup_fn(inputs[0], labels)
            inputs[0] = samples
        
        # for sure the current model contains no grad
        optimizer.zero_grad()

        # interpolate models
        
        if cfg.TRAIN.LINEAR_CONNECT_CLIMB:
            cur_weights = copy.deepcopy(model.state_dict())
            
            assert(len(cur_weights) == len(prev_weights))
            interpolate_weights = {}
            assert(cfg.TRAIN.LINEAR_CONNECT_SAMPLE_L < cfg.TRAIN.LINEAR_CONNECT_SAMPLE_R)
            patch_ratio = random.uniform(cfg.TRAIN.LINEAR_CONNECT_SAMPLE_L, cfg.TRAIN.LINEAR_CONNECT_SAMPLE_R)
            # patch_ratio = 0.5

            # for safety, no grads should be transferred.
            with torch.no_grad():
                for key in prev_weights:
                    # interpolate_weights['module.model.'+key] = prev_weights[key]
                    interpolate_weights['module.model.'+key] = prev_weights[key] * patch_ratio + cur_weights['module.model.'+key] * (1 - patch_ratio)
                model.load_state_dict(interpolate_weights)
                # model.load_state_dict(interpolate_weights, strict=False)

            with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
                # Explicitly declare reduction to mean.
                perform_backward = True
                optimizer.zero_grad()

                preds = model(inputs)
                
                # Compute the loss.
                loss = loss_fun(preds, labels)
                loss = loss * cfg.TRAIN.LINEAR_CONNECT_LOSS_RATIO

            if perform_backward:
                scaler.scale(loss).backward()
            # Unscales the gradients of optimizer's assigned params in-place
            # scaler.unscale_(optimizer)
            # achieve param grads
            grads_record = {}
            for name, params in model.named_parameters():
                if params.grad != None:
                    # grads_record[name] = params.grad.clone().detach() * (1-patch_ratio)
                    grads_record[name] = params.grad.clone().detach()
                else:
                    grads_record[name] = params.grad
            grads_record = copy.deepcopy(grads_record)
            
            # restore model weights
            # for safety
            with torch.no_grad():
                model.load_state_dict(cur_weights)
        
        # normal training  
        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):

            # Explicitly declare reduction to mean.
            perform_backward = True
            optimizer.zero_grad()
            
            if cfg.MODEL.MODEL_NAME == "ContrastiveModel":
                (
                    model,
                    preds,
                    partial_loss,
                    perform_backward,
                ) = contrastive_forward(
                    model, cfg, inputs, index, time, epoch_exact, scaler
                )
            elif cfg.DETECTION.ENABLE:
                # Compute the predictions.
                preds = model(inputs, meta["boxes"])
            elif cfg.MASK.ENABLE:
                preds, labels = model(inputs)
            else:
                if cfg.MODEL.RECORD_ROUTING:
                    preds, rout_state = model(inputs)
                    
                    ori_clip_freq = rout_state[:,:,:,0]
                    # dis = torch.clamp(cfg.MODEL.ROUTING_FREQUENCE_CONSTRAIN - ori_clip_freq, min=0.0)
                    dis = cfg.MODEL.ROUTING_FREQUENCE_CONSTRAIN - ori_clip_freq
                    if cfg.MODEL.LOSS_FREQ_TYPE == "mse":
                        loss_freq = (dis ** 2).mean()
                    elif cfg.MODEL.LOSS_FREQ_TYPE == "hinge":
                        loss_freq = torch.clamp(dis, min=0).mean()
                    else:
                        raise ValueError ("Invalid loss_freq type: ", cfg.MODEL.LOSS_FREQ_TYPE)
                
                elif cfg.MODEL.KEEP_RAW_MODEL and cfg.MODEL.RAW_MODEL_DISTILLATION:
                    preds, raw_preds = model(inputs)
                
                else:
                    preds = model(inputs)
            
            if cfg.TASK == "ssl" and cfg.MODEL.MODEL_NAME == "ContrastiveModel":
                labels = torch.zeros(
                    preds.size(0), dtype=labels.dtype, device=labels.device
                )

            if cfg.MODEL.MODEL_NAME == "ContrastiveModel" and partial_loss:
                loss = partial_loss
            else:
                # Compute the loss.
                loss = loss_fun(preds, labels)
                
                if cfg.MODEL.RECORD_ROUTING:
                    ori_loss = loss
                    loss = cfg.MODEL.CLS_LOSS_RATIO * loss + cfg.MODEL.ROUTING_FREQ_CONS_FACTOR * loss_freq
                    if cur_iter % cfg.LOG_PERIOD == 0:
                        print('Routing average choose clip weight ratio:%.4f'%ori_clip_freq.mean().item())
                        print('Cls loss:%.4f'%ori_loss.item())
                        print('Freq loss:%.4f'%loss_freq.item())
                        print('Freq loss factor:%f'%cfg.MODEL.ROUTING_FREQ_CONS_FACTOR)
                        print('Routing average choose clip weight ratio each router: ')
                        print(ori_clip_freq.mean(-1).mean(-1).detach().cpu().numpy())
                        print('\n')
                
                if cfg.MODEL.KEEP_RAW_MODEL and cfg.MODEL.RAW_MODEL_DISTILLATION:
                    T = 1.0
                    distillation_loss = F.kl_div(
                        F.log_softmax(preds / T, dim=1),
                        F.log_softmax(raw_preds / T, dim=1),
                        reduction='sum',
                        log_target=True
                    ) * (T * T) / preds.numel()
                    
                    if cur_iter % cfg.LOG_PERIOD == 0:
                        logger.info('Distillation Loss: %.8f'%distillation_loss.item())
                        logger.info('Distillation Loss Ratio: %f'%cfg.MODEL.DISTILLATION_RATIO)

                    loss += cfg.MODEL.DISTILLATION_RATIO * distillation_loss
                    
                if cfg.TRAIN.EWC_SET:
                    if (cfg.TRAIN.ZS_RESTART_CONS and cfg.TRAIN.ZS_RESTART_EPOCH != -1 and (cur_epoch-cfg.SOLVER.WARMUP_EPOCHS+1)%cfg.TRAIN.ZS_RESTART_EPOCH==0 and (cur_epoch-cfg.SOLVER.WARMUP_EPOCHS+1 > 0)) or cfg.TRAIN.ZS_RESTART_CONS == False:
                    
                        loss_reg = 0
                        
                        for n, p in model.named_parameters():
                            # maybe ignore module.model.logit_scale ?
                            rawclip_name = None 
                            if 'module.model' in n:
                                rawclip_name = n.replace('model', 'raw_model')
                            else:
                                continue
                            """ 
                            if cfg.TRAIN.EWC_IDENTITY_FISHER:
                                loss_reg += torch.sum((p - raw_clip_params[rawclip_name]).pow(2)) / 2

                            else:
                            """
                            if n in fisher_map:
                                loss_reg += torch.sum(fisher_map[n] * (p - raw_clip_params[rawclip_name]).pow(2)) / 2
                        if cur_iter % cfg.LOG_PERIOD == 0:
                            logger.info('Reg Loss: %.8f'%loss_reg)
                            logger.info('Reg Loss Ratio: %f'%cfg.TRAIN.EWC_CONSTRAIN_RATIO)

                        loss += cfg.TRAIN.EWC_CONSTRAIN_RATIO * loss_reg
                

        loss_extra = None
        if isinstance(loss, (list, tuple)):
            loss, loss_extra = loss

        # check Nan Loss.
        misc.check_nan_losses(loss)
        if perform_backward:
            scaler.scale(loss).backward()
        
        # if linear-connect-climb, then do the gradients merging
        if cfg.TRAIN.LINEAR_CONNECT_CLIMB:
            state_dict = model.state_dict(keep_vars=True)
            for param_name in state_dict.keys():
                if state_dict[param_name].grad == None:
                    continue
                state_dict[param_name].grad += grads_record[param_name]
                # state_dict[param_name].grad += 0

            model.load_state_dict(state_dict)
        
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)

        # Clip gradients if necessary
        if cfg.SOLVER.CLIP_GRAD_VAL:
            grad_norm = torch.nn.utils.clip_grad_value_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL
            )
        elif cfg.SOLVER.CLIP_GRAD_L2NORM:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
            )
        else:
            grad_norm = optim.get_grad_norm_(model.parameters())
        # Update the parameters. (defaults to True)
        model, update_param = contrastive_parameter_surgery(
            model, cfg, epoch_exact, cur_iter
        )
        if update_param:
            scaler.step(optimizer)
        scaler.update()

        if cfg.MIXUP.ENABLE:
            _top_max_k_vals, top_max_k_inds = torch.topk(
                labels, 2, dim=1, largest=True, sorted=True
            )
            idx_top1 = torch.arange(labels.shape[0]), top_max_k_inds[:, 0]
            idx_top2 = torch.arange(labels.shape[0]), top_max_k_inds[:, 1]
            preds = preds.detach()
            preds[idx_top1] += preds[idx_top2]
            preds[idx_top2] = 0.0
            labels = top_max_k_inds[:, 0]

        if cfg.DETECTION.ENABLE:
            if cfg.NUM_GPUS > 1:
                loss = du.all_reduce([loss])[0]
            loss = loss.item()

            # Update and log stats.
            train_meter.update_stats(None, None, None, loss, lr)
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Train/loss": loss, "Train/lr": lr},
                    global_step=data_size * cur_epoch + cur_iter,
                )

        else:
            top1_err, top5_err = None, None
            if cfg.DATA.MULTI_LABEL:
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, grad_norm = du.all_reduce([loss, grad_norm])
                loss, grad_norm = (
                    loss.item(),
                    grad_norm.item(),
                )
            elif cfg.MASK.ENABLE:
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, grad_norm = du.all_reduce([loss, grad_norm])
                    if loss_extra:
                        loss_extra = du.all_reduce(loss_extra)
                loss, grad_norm, top1_err, top5_err = (
                    loss.item(),
                    grad_norm.item(),
                    0.0,
                    0.0,
                )
                if loss_extra:
                    loss_extra = [one_loss.item() for one_loss in loss_extra]
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, grad_norm, top1_err, top5_err = du.all_reduce(
                        [loss.detach(), grad_norm, top1_err, top5_err]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, grad_norm, top1_err, top5_err = (
                    loss.item(),
                    grad_norm.item(),
                    top1_err.item(),
                    top5_err.item(),
                )

            # Update and log stats.
            train_meter.update_stats(
                top1_err,
                top5_err,
                loss,
                lr,
                grad_norm,
                batch_size
                * max(
                    cfg.NUM_GPUS, 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                loss_extra,
            )
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {
                        "Train/loss": loss,
                        "Train/lr": lr,
                        "Train/Top1_err": top1_err,
                        "Train/Top5_err": top5_err,
                    },
                    global_step=data_size * cur_epoch + cur_iter,
                )
        train_meter.iter_toc()  # do measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        torch.cuda.synchronize()
        train_meter.iter_tic()
    del inputs

    # in case of fragmented memory
    torch.cuda.empty_cache()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(
    val_loader, model, val_meter, cur_epoch, cfg, train_loader, writer
):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, labels, index, time, meta) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
            index = index.cuda()
            time = time.cuda()
        batch_size = (
            inputs[0][0].size(0)
            if isinstance(inputs[0], list)
            else inputs[0].size(0)
        )
        val_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            if cfg.NUM_GPUS:
                preds = preds.cpu()
                ori_boxes = ori_boxes.cpu()
                metadata = metadata.cpu()

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(preds, ori_boxes, metadata)

        else:
            if cfg.TASK == "ssl" and cfg.MODEL.MODEL_NAME == "ContrastiveModel":
                if not cfg.CONTRASTIVE.KNN_ON:
                    return
                train_labels = (
                    model.module.train_labels
                    if hasattr(model, "module")
                    else model.train_labels
                )
                yd, yi = model(inputs, index, time)
                K = yi.shape[1]
                C = (
                    cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM
                )  # eg 400 for Kinetics400
                candidates = train_labels.view(1, -1).expand(batch_size, -1)
                retrieval = torch.gather(candidates, 1, yi)
                retrieval_one_hot = torch.zeros((batch_size * K, C)).cuda()
                retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
                yd_transform = yd.clone().div_(cfg.CONTRASTIVE.T).exp_()
                probs = torch.mul(
                    retrieval_one_hot.view(batch_size, -1, C),
                    yd_transform.view(batch_size, -1, 1),
                )
                preds = torch.sum(probs, 1)
            else:
                if cfg.MODEL.RECORD_ROUTING:
                    preds, rout_state = model(inputs)
                elif cfg.MODEL.KEEP_RAW_MODEL and cfg.MODEL.RAW_MODEL_DISTILLATION:
                    preds, raw_preds = model(inputs)
                else:
                    preds = model(inputs)

                # preds = model(inputs)

            if cfg.DATA.MULTI_LABEL:
                if cfg.NUM_GPUS > 1:
                    preds, labels = du.all_gather([preds, labels])
            else:
                if cfg.DATA.IN22k_VAL_IN1K != "":
                    preds = preds[:, :1000]
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

                # Combine the errors across the GPUs.
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                if cfg.NUM_GPUS > 1:
                    top1_err, top5_err = du.all_reduce([top1_err, top5_err])

                # Copy the errors from GPU to CPU (sync point).
                top1_err, top5_err = top1_err.item(), top5_err.item()

                val_meter.iter_toc()
                # Update and log stats.
                val_meter.update_stats(
                    top1_err,
                    top5_err,
                    batch_size
                    * max(
                        cfg.NUM_GPUS, 1
                    ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                )
                # write to tensorboard format if available.
                if writer is not None:
                    writer.add_scalars(
                        {"Val/Top1_err": top1_err, "Val/Top5_err": top5_err},
                        global_step=len(val_loader) * cur_epoch + cur_iter,
                    )

            val_meter.update_predictions(preds, labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
        if cfg.DETECTION.ENABLE:
            writer.add_scalars(
                {"Val/mAP": val_meter.full_map}, global_step=cur_epoch
            )
        else:
            all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
            all_labels = [
                label.clone().detach() for label in val_meter.all_labels
            ]
            if cfg.NUM_GPUS:
                all_preds = [pred.cpu() for pred in all_preds]
                all_labels = [label.cpu() for label in all_labels]
            writer.plot_eval(
                preds=all_preds, labels=all_labels, global_step=cur_epoch
            )

    val_meter.reset()


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def build_trainer(cfg):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    model = build_model(cfg)
    
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        flops, params = misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    
    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(
        cfg, "train", is_precise_bn=True
    )
    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    return (
        model,
        optimizer,
        train_loader,
        val_loader,
        precise_bn_loader,
        train_meter,
        val_meter,
    )


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    try:
        du.init_distributed_training(cfg)
    except:
        du.init_distributed_training(cfg.NUM_GPUS, cfg.SHARD_ID)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))
    
    # Build the video model and print model statistics.
    model = build_model(cfg)

    # custom load checkpoint here
    if cfg.TRAIN.CUSTOM_LOAD: 
        custom_load_file = cfg.TRAIN.CUSTOM_LOAD_FILE
        assert pathmgr.exists(
                custom_load_file
                ), "Checkpoint '{}' not found".format(custom_load_file)
        logger.info("Loading custom network weights from {}.".format(custom_load_file))
        checkpoint = torch.load(custom_load_file, map_location='cpu')
        checkpoint_model = checkpoint['model_state']
        state_dict = model.state_dict() 
        
        if cfg.VAL_MODE and cfg.TEST.PATCHING_MODEL and cfg.TEST.CLIP_ORI_PATH:
            logger.info("patching model")
            patching_ratio = cfg.TEST.PATCHING_RATIO
            try:
                clip_ori_state = torch.jit.load(cfg.TEST.CLIP_ORI_PATH, map_location='cpu').state_dict()
                # pop some unnessesary keys
                _ = [clip_ori_state.pop(i) for i in ['input_resolution', 'context_length', 'vocab_size']]
                raw_clip_flag = True
            except:
                clip_ori_state = torch.load(cfg.TEST.CLIP_ORI_PATH, map_location='cpu')['model_state']
                raw_clip_flag = False

            logger.info("model contains %d keys for patching"%len(checkpoint_model))
            logger.info("original clip model contains %d keys"%len(clip_ori_state))

            missing_params_name = None
            if len(clip_ori_state) == len(checkpoint_model):
                logger.info("no extra params added")
            else:
                if raw_clip_flag:
                    logger.info("Missing Params for patching:")
                    logger.info(list(set(checkpoint_model.keys())-set(['model.'+i for i in clip_ori_state.keys()])))
                    missing_params_name = list(set(checkpoint_model.keys())-set(['model.'+i for i in clip_ori_state.keys()]))
                else:
                    missing_params_name = list(set(checkpoint_model.keys())-set([i for i in clip_ori_state.keys()]))


            # add model prefix
            patching_checkpoint_model = {}
            for key in clip_ori_state: 
                if raw_clip_flag:
                    patching_checkpoint_model['model.'+key] = clip_ori_state[key] * cfg.TEST.PATCHING_RATIO + checkpoint_model['model.'+key] * (1 - cfg.TEST.PATCHING_RATIO)
                else:
                    if key not in checkpoint_model:
                        continue
                    patching_checkpoint_model[key] = clip_ori_state[key] * cfg.TEST.PATCHING_RATIO + checkpoint_model[key] * (1 - cfg.TEST.PATCHING_RATIO)

            if missing_params_name != None:
                for key in missing_params_name:
                    patching_checkpoint_model[key] = checkpoint_model[key]

            checkpoint_model = patching_checkpoint_model

        if 'module' in list(state_dict.keys())[0]:
            new_checkpoint_model = {} 
            for key, value in checkpoint_model.items(): 
                new_checkpoint_model['module.' + key] = value
                # new_checkpoint_model['module.' + key.replace('model', 'raw_model')] = value
            checkpoint_model = new_checkpoint_model

        for key in checkpoint_model.keys():
            logger.info("missing some parameters")
            if key not in state_dict.keys():
                logger.info(key)
        
        model.load_state_dict(checkpoint_model, strict=False) 

    flops, params = 0.0, 0.0
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        flops, params = misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    """ 
    for m_name, m in model.module.named_modules():
        for p_name, p in m.named_parameters(recurse=False):
            if p.requires_grad == True:
                print("==")
                print(m_name)
                print(p_name)
    import time; time.sleep(20)
    exit()
    """
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)
    # Load a checkpoint to resume training if applicable.
    if cfg.TRAIN.AUTO_RESUME and cu.has_checkpoint(cfg.OUTPUT_DIR):
        logger.info("Load from last checkpoint.")
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR, task=cfg.TASK)
        if last_checkpoint is not None:
            checkpoint_epoch = cu.load_checkpoint(
                last_checkpoint,
                model,
                cfg.NUM_GPUS > 1,
                optimizer,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
            start_epoch = checkpoint_epoch + 1
        elif "ssl_eval" in cfg.TASK:
            last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR, task="ssl")
            checkpoint_epoch = cu.load_checkpoint(
                last_checkpoint,
                model,
                cfg.NUM_GPUS > 1,
                optimizer,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
                epoch_reset=True,
                clear_name_pattern=cfg.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN,
            )
            start_epoch = checkpoint_epoch + 1
        else:
            start_epoch = 0
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        logger.info("Load from given checkpoint file.")
        checkpoint_epoch = cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            optimizer,
            scaler if cfg.TRAIN.MIXED_PRECISION else None,
            inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
            epoch_reset=cfg.TRAIN.CHECKPOINT_EPOCH_RESET,
            clear_name_pattern=cfg.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN,
            image_init=cfg.TRAIN.CHECKPOINT_IN_INIT,
        )
        start_epoch = checkpoint_epoch + 1
    else:
        start_epoch = 0

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    
    raw_batch_size = cfg.TRAIN.BATCH_SIZE
    raw_mixup = cfg.MIXUP.ENABLE
    # raw_aug_num_sample = cfg.AUG.NUM_SAMPLE
    cfg.TRAIN.BATCH_SIZE = cfg.TRAIN.BATCH_SIZE // 2
    cfg.MIXUP.ENABLE = False
    # cfg.AUG.NUM_SAMPLE = 1
    fisher_loader = loader.construct_loader(cfg, "train")
    cfg.TRAIN.BATCH_SIZE = raw_batch_size
    cfg.MIXUP.ENABLE = raw_mixup
    # cfg.AUG.NUM_SAMPLE = raw_aug_num_sample

    precise_bn_loader = (
        loader.construct_loader(cfg, "train", is_precise_bn=True)
        if cfg.BN.USE_PRECISE_STATS
        else None
    )

    if (
        cfg.TASK == "ssl"
        and cfg.MODEL.MODEL_NAME == "ContrastiveModel"
        and cfg.CONTRASTIVE.KNN_ON
    ):
        if hasattr(model, "module"):
            model.module.init_knn_labels(train_loader)
        else:
            model.init_knn_labels(train_loader)

    # Create meters.
    if cfg.DETECTION.ENABLE:
        train_meter = AVAMeter(len(train_loader), cfg, mode="train")
        val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    else:
        train_meter = TrainMeter(len(train_loader), cfg)
        val_meter = ValMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))
    
    if cfg.VAL_MODE:
        logger.info("Only do validation")
        eval_epoch(
                val_loader,
                model,
                val_meter,
                0,
                cfg,
                train_loader,
                writer,
            )
        return 

    epoch_timer = EpochTimer()
    
    if cfg.TRAIN.EWC_SET:
        if cfg.TRAIN.EWC_LOAD_FILE:
            fisher_map = torch.load(cfg.TRAIN.EWC_LOAD_FILE, map_location='cpu')
            # fisher_map = fisher_map.to(model.device)
            for key in fisher_map:
                fisher_map[key] = fisher_map[key].to(model.device)
                
        # elif cfg.TRAIN.EWC_IDENTITY_FISHER:
        #     fisher_map = None
        
        else:
            fisher_map = compute_fisher_matrix_diag(
                    fisher_loader,
                    model,
                    optimizer,
                    scaler,
                    train_meter,
                    0,
                    cfg,
                    writer,
                )
        del fisher_loader

    else:
        fisher_map = None
    # maybe ignore the module.model.logit_scale fisher value

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):

        if cur_epoch > 0 and cfg.DATA.LOADER_CHUNK_SIZE > 0:
            num_chunks = math.ceil(
                cfg.DATA.LOADER_CHUNK_OVERALL_SIZE / cfg.DATA.LOADER_CHUNK_SIZE
            )
            skip_rows = (cur_epoch) % num_chunks * cfg.DATA.LOADER_CHUNK_SIZE
            logger.info(
                f"=================+++ num_chunks {num_chunks} skip_rows {skip_rows}"
            )
            cfg.DATA.SKIP_ROWS = skip_rows
            logger.info(f"|===========| skip_rows {skip_rows}")
            train_loader = loader.construct_loader(cfg, "train")
            loader.shuffle_dataset(train_loader, cur_epoch)

        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
            if changed:
                (
                    model,
                    optimizer,
                    train_loader,
                    val_loader,
                    precise_bn_loader,
                    train_meter,
                    val_meter,
                ) = build_trainer(cfg)

                # Load checkpoint.
                if cu.has_checkpoint(cfg.OUTPUT_DIR):
                    last_checkpoint = cu.get_last_checkpoint(
                        cfg.OUTPUT_DIR, task=cfg.TASK
                    )
                    assert "{:05d}.pyth".format(cur_epoch) in last_checkpoint
                else:
                    last_checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
                logger.info("Load from {}".format(last_checkpoint))
                cu.load_checkpoint(
                    last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
                )

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)
        if hasattr(train_loader.dataset, "_set_epoch_num"):
            train_loader.dataset._set_epoch_num(cur_epoch)
        # Train for one epoch.
        epoch_timer.epoch_tic()
        # If do the zs_constrain
        if (cfg.TRAIN.ZS_CONS 
                or (cfg.TRAIN.ZS_INIT_CONS and cur_epoch == 0) 
                or (cfg.TRAIN.ZS_RESTART_CONS and cfg.TRAIN.ZS_RESTART_EPOCH != -1 and (cur_epoch-cfg.SOLVER.WARMUP_EPOCHS+1)%cfg.TRAIN.ZS_RESTART_EPOCH==0 and (cur_epoch-cfg.SOLVER.WARMUP_EPOCHS+1 > 0))
                ) \
            and cfg.TRAIN.CLIP_ORI_PATH:

            logger.info("Constrain Model Parameter Change Per Epoch")
            
            # current model ckpt
            state_dict = model.state_dict() 

            # latest checkpoint params, almost the same as "state_dict"
            # checkpoint = torch.load(os.path.join(cfg.OUTPUT_DIR, 'checkpoints', "checkpoint_epoch_{:05d}.pyth".format(-1)), map_location='cpu')
            # checkpoint_model = checkpoint['model_state']
            checkpoint_model = model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict()
            checkpoint_model = cu.sub_to_normal_bn(checkpoint_model)
            logger.info("model contains %d keys for patching"%len(checkpoint_model))
            
            try:
                # zero shot model
                clip_ori_state = torch.jit.load(cfg.TRAIN.CLIP_ORI_PATH, map_location=model.device).state_dict()
                # pop some unnessesary keys
                _ = [clip_ori_state.pop(i) for i in ['input_resolution', 'context_length', 'vocab_size']]
                raw_clip_flag = True
            except:
                clip_ori_state = torch.load(cfg.TRAIN.CLIP_ORI_PATH, map_location=model.device)['model_state']
                raw_clip_flag = False

            logger.info("model contains %d keys for patching"%len(checkpoint_model))
            clip_model_keys = clip_ori_state.keys()
            new_model_keys = checkpoint_model.keys()
            
            if cfg.MODEL.NUM_EXPERTS > 0:
                for key in list(clip_ori_state.keys()):
                    if 'mlp' in key and key.startswith('visual'):
                        layer_id = int(key.split('.mlp')[0].split('.')[-1])
                        if layer_id not in cfg.MODEL.EXPERT_INSERT_LAYERS:
                            continue
                        for expert_id in range(cfg.MODEL.NUM_EXPERTS):
                            if 'c_fc' in key or 'gelu' in key:
                                new_key = key.replace('mlp', 'experts_head.%d'%expert_id)
                            else:
                                new_key = key.replace('mlp', 'experts_tail.%d'%expert_id)
                            clip_ori_state[new_key] = clip_ori_state[key]

                    logger.info("expanded original clip model contains %d keys"%len(clip_ori_state))
            
            missing_params_name = None
            if len(clip_ori_state) == len(checkpoint_model):
                logger.info("no extra params added")
            else:
                if raw_clip_flag:
                    logger.info("Missing Params for patching:")
                    logger.info(list(set(checkpoint_model.keys())-set(['model.'+i for i in clip_ori_state.keys()])))
                    missing_params_name = list(set(checkpoint_model.keys())-set(['model.'+i for i in clip_ori_state.keys()]))
                else:
                    missing_params_name = list(set(checkpoint_model.keys())-set([i for i in clip_ori_state.keys()]))

            # add model prefix
            patching_checkpoint_model = {}
            if cfg.TRAIN.ADAPT_ZS_CONS_RATIO:
                zs_scale = optim.get_epoch_lr(cur_epoch, cfg) / cfg.SOLVER.BASE_LR
                patching_ratio = cfg.TRAIN.ZS_CONS_RATIO * zs_scale
            else:
                patching_ratio = cfg.TRAIN.ZS_CONS_RATIO
             
            for key in clip_ori_state: 
                if raw_clip_flag:
                    patching_checkpoint_model['model.'+key] = clip_ori_state[key] * patching_ratio + checkpoint_model['model.'+key] * (1 - patching_ratio)
                    patching_checkpoint_model['raw_model.'+key] = clip_ori_state[key] * patching_ratio + checkpoint_model['model.'+key] * (1 - patching_ratio)
                else:
                    if key not in checkpoint_model:
                        continue

                    patching_checkpoint_model[key] = clip_ori_state[key] * patching_ratio + checkpoint_model[key] * (1 - patching_ratio)
                    if 'model' in key and 'raw_model' not in key:
                        patching_checkpoint_model[key.replace('model', 'raw_model', 1)] = clip_ori_state[key] * patching_ratio + checkpoint_model[key] * (1 - patching_ratio)
            
            if missing_params_name != None:
                for key in missing_params_name:
                    if 'raw_model' not in key:
                        patching_checkpoint_model[key] = checkpoint_model[key]
            
            checkpoint_model = patching_checkpoint_model
            
            if 'module' in list(state_dict.keys())[0]:
                new_checkpoint_model = {} 
                for key, value in checkpoint_model.items(): 
                    new_checkpoint_model['module.' + key] = value
                checkpoint_model = new_checkpoint_model

            for key in checkpoint_model.keys():
                if key not in state_dict.keys():
                    print(key)

            model.load_state_dict(checkpoint_model, strict=False) 
        
               
        train_epoch(
            train_loader,
            model,
            optimizer,
            scaler,
            train_meter,
            cur_epoch,
            cfg,
            writer,
            fisher_map,
        )
        epoch_timer.epoch_toc()
        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
            f"For epoch {cur_epoch}, each iteraction takes "
            f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
        )

        is_checkp_epoch = (
            cu.is_checkpoint_epoch(
                cfg,
                cur_epoch,
                None if multigrid is None else multigrid.schedule,
            )
            or cur_epoch == cfg.SOLVER.MAX_EPOCH - 1
        )
        is_eval_epoch = (
            misc.is_eval_epoch(
                cfg,
                cur_epoch,
                None if multigrid is None else multigrid.schedule,
            )
            and not cfg.MASK.ENABLE
        )
        
        """
        # Compute precise BN stats.
        if (
            (is_checkp_epoch or is_eval_epoch)
            and cfg.BN.USE_PRECISE_STATS
            and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)
        """
        if len(get_bn_modules(model)) and cfg.BN.USE_PRECISE_STATS:
            calculate_and_update_precise_bn(
                    precise_bn_loader,
                    model,
                    min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                    cfg.NUM_GPUS > 0,
                )
        _ = misc.aggregate_sub_bn_stats(model)
        """
        # cover the former latest checkpoint
        cu.save_checkpoint(
            cfg.OUTPUT_DIR,
            model,
            optimizer,
            -2,
            cfg,
            scaler if cfg.TRAIN.MIXED_PRECISION else None,
        )
        """

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(
                cfg.OUTPUT_DIR,
                model,
                optimizer,
                cur_epoch,
                cfg,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
        # Evaluate the model on validation set.
        if is_eval_epoch:
            eval_epoch(
                val_loader,
                model,
                val_meter,
                cur_epoch,
                cfg,
                train_loader,
                writer,
            )
    if start_epoch == cfg.SOLVER.MAX_EPOCH: # eval if we loaded the final checkpoint
        eval_epoch(val_loader, model, val_meter, start_epoch, cfg, train_loader, writer)
    if writer is not None:
        writer.close()
    result_string = (
        "_p{:.2f}_f{:.2f} _t{:.2f}_m{:.2f} _a{:.2f} Top5 Acc: {:.2f} MEM: {:.2f} f: {:.4f}"
        "".format(
            params / 1e6,
            flops,
            epoch_timer.median_epoch_time() / 60.0
            if len(epoch_timer.epoch_times)
            else 0.0,
            misc.gpu_mem_usage(),
            100 - val_meter.min_top1_err,
            100 - val_meter.min_top5_err,
            misc.gpu_mem_usage(),
            flops,
        )
    )
    logger.info("training done: {}".format(result_string))

    return result_string
