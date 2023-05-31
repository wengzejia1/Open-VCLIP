#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch
from pytorchvideo.layers.distributed import get_local_rank

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.env import pathmgr
from slowfast.utils.meters import AVAMeter, TestMeter
from slowfast.utils.env import pathmgr

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()
    
    if cfg.MODEL.RECORD_ROUTING:
        rout_list = []

    for cur_iter, (inputs, labels, video_idx, time, meta) in enumerate(
        test_loader
    ):
       
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            video_idx = video_idx.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        test_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
            ori_boxes = (
                ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
            )
            metadata = (
                metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()
            )

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(preds, ori_boxes, metadata)
            test_meter.log_iter_stats(None, cur_iter)
        elif cfg.TASK == "ssl" and cfg.MODEL.MODEL_NAME == "ContrastiveModel":
            if not cfg.CONTRASTIVE.KNN_ON:
                test_meter.finalize_metrics()
                return test_meter
            # preds = model(inputs, video_idx, time)
            train_labels = (
                model.module.train_labels
                if hasattr(model, "module")
                else model.train_labels
            )
            yd, yi = model(inputs, video_idx, time)
            batchSize = yi.shape[0]
            K = yi.shape[1]
            C = cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM  # eg 400 for Kinetics400
            candidates = train_labels.view(1, -1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)
            retrieval_one_hot = torch.zeros((batchSize * K, C)).cuda()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(cfg.CONTRASTIVE.T).exp_()
            probs = torch.mul(
                retrieval_one_hot.view(batchSize, -1, C),
                yd_transform.view(batchSize, -1, 1),
            )
            preds = torch.sum(probs, 1)
        else:
            if (cfg.MODEL.KEEP_RAW_MODEL and cfg.MODEL.ENSEMBLE_PRED) and cfg.MODEL.RECORD_ROUTING:
                print("ensemble pred should not exists together with record_routing")
                exit()

            # Perform the forward pass.
            if cfg.MODEL.RECORD_ROUTING:
                preds, routing_state = model(inputs)
                # routing_state shape [layer_num, patch_num, bz * clip_len, 2)
                rshape = routing_state.shape
                routing_state = routing_state.reshape(rshape[0], rshape[1], inputs[0].shape[0], -1, 2).permute(2, 0, 1, 3, 4)
                if get_local_rank() == 0:
                    if cur_iter % 10 == 0:
                        print(routing_state[:,:,:,:,0].mean(-1).mean(0).detach().cpu().squeeze().numpy())
            
            elif cfg.MODEL.KEEP_RAW_MODEL and cfg.MODEL.ENSEMBLE_PRED:
                preds, raw_preds = model(inputs)
                preds = cfg.MODEL.ENSEMBLE_RAWMODEL_RATIO * raw_preds + (1 - cfg.MODEL.ENSEMBLE_RAWMODEL_RATIO) * preds

            else:
                preds = model(inputs)
        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            preds, labels, video_idx = du.all_gather([preds, labels, video_idx])
            """
            if cfg.MODEL.RECORD_ROUTING:
                routing_state = du.all_gather([routing_state])[0]
                routing_state = routing_state.cpu()
                rout_list.append(routing_state)
            """
            # if cfg.MODEL.RECORD_ROUTING and cur_iter >= 10:
            #     break

        if cfg.NUM_GPUS:
            preds = preds.cpu()
            labels = labels.cpu()
            video_idx = video_idx.cpu()

        test_meter.iter_toc()

        if not cfg.VIS_MASK.ENABLE:
            # Update and log stats.
            test_meter.update_stats(
                preds.detach(), labels.detach(), video_idx.detach()
            )
        test_meter.log_iter_stats(cur_iter) 
        test_meter.iter_tic()
    
    # routing record verify
    """
    if cfg.MODEL.RECORD_ROUTING:
        if get_local_rank() == 0: 
            rout_record = torch.cat(rout_list, 0)
            torch.save(rout_record, "%s/%s_rout_record.pth"%(cfg.OUTPUT_DIR, cfg.DATA.PATH_TO_DATA_DIR.split('/')[-1]))
    """ 

    # Log epoch stats and print the final testing results.
    if not cfg.DETECTION.ENABLE:
        all_preds = test_meter.video_preds.clone().detach()
        all_labels = test_meter.video_labels
        if cfg.NUM_GPUS:
            all_preds = all_preds.cpu()
            all_labels = all_labels.cpu()
        if writer is not None:
            writer.plot_eval(preds=all_preds, labels=all_labels)

        if cfg.TEST.SAVE_RESULTS_PATH != "":
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

            if du.is_root_proc():
                with pathmgr.open(save_path, "wb") as f:
                    pickle.dump([all_preds, all_labels], f)

            logger.info(
                "Successfully saved prediction results to {}".format(save_path)
            )

        if False:
            all_preds = test_meter.video_preds.clone().detach()
            all_labels = test_meter.video_labels
            accumulate = {}
            for idx in range(len(all_labels)):
                label = int(all_labels[idx])
                if label not in accumulate:
                    accumulate[label] = []
                if torch.argmax(all_preds[idx], 0) == label:
                    accumulate[label].append(1)
                else:
                    accumulate[label].append(0)
            
            # find the half most classes
            name = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, 'train.csv')
            cls_freq = {}
            with open(name, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    cls = int(line.split(",")[1])
                    if cls not in cls_freq:
                        cls_freq[cls] = 0
                    cls_freq[cls] += 1

            cls_freq_list = []
            for cls_id in range(len(cls_freq)):
                cls_freq_list.append((cls_id,cls_freq[cls_id]))
            cls_freq_list = sorted(cls_freq_list, key = lambda x:x[1], reverse=True)
            closeset = [i[0] for i in cls_freq_list[:200]]
            openset = [i[0] for i in cls_freq_list[200:]]
             
            print(len(accumulate))
            print(len(closeset))
            print(len(openset))

            closeset_acc = []
            openset_acc = []
            for label in closeset:
                closeset_acc += accumulate[label]
            for label in openset:
                openset_acc += accumulate[label]

            openset_acc = sum(openset_acc) / len(openset_acc)
            closeset_acc = sum(closeset_acc) / len(closeset_acc)
            
            print('top-1 closeset acc: %f'%(closeset_acc))
            print('top-1 openset acc: %f'%(openset_acc))

    test_meter.finalize_metrics()
    return test_meter


def test(cfg):

    """
    Perform multi-view testing on the pretrained video model.
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

    if len(cfg.TEST.NUM_TEMPORAL_CLIPS) == 0:
        cfg.TEST.NUM_TEMPORAL_CLIPS = [cfg.TEST.NUM_ENSEMBLE_VIEWS]

    test_meters = []
    for num_view in cfg.TEST.NUM_TEMPORAL_CLIPS:

        cfg.TEST.NUM_ENSEMBLE_VIEWS = num_view

        # Print config.
        logger.info("Test with config:")
        logger.info(cfg)
        # Build the video model and print model statistics.
        model = build_model(cfg)
         
        if not cfg.TEST.CUSTOM_LOAD:
            cu.load_test_checkpoint(cfg, model)

        # custom load checkpoint here
        if cfg.TEST.CUSTOM_LOAD:
            custom_load_file = cfg.TEST.CUSTOM_LOAD_FILE
            assert pathmgr.exists(
                    custom_load_file
            ), "Checkpoint '{}' not found".format(custom_load_file)
            logger.info("Loading custom network weights from {}.".format(custom_load_file)) 
            checkpoint = torch.load(custom_load_file, map_location='cpu')
            checkpoint_model = checkpoint['model_state']
            state_dict = model.state_dict()
             
            if cfg.TEST.PATCHING_MODEL and cfg.TEST.CLIP_ORI_PATH:
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
                checkpoint_model = new_checkpoint_model
            
            for key in checkpoint_model.keys(): 
                if key not in state_dict.keys():
                    logger.info("missing parameters")
                    logger.info(key)
            
            model.load_state_dict(checkpoint_model, strict=False)

        flops, params = 0.0, 0.0
        if du.is_master_proc() and cfg.LOG_MODEL_INFO:
            model.eval()
            flops, params = misc.log_model_info(
                model, cfg, use_train_input=False
            )

        if du.is_master_proc() and cfg.LOG_MODEL_INFO:
            misc.log_model_info(model, cfg, use_train_input=False)
        if (
            cfg.TASK == "ssl"
            and cfg.MODEL.MODEL_NAME == "ContrastiveModel"
            and cfg.CONTRASTIVE.KNN_ON
        ):
            train_loader = loader.construct_loader(cfg, "train")
            if hasattr(model, "module"):
                model.module.init_knn_labels(train_loader)
            else:
                model.init_knn_labels(train_loader)
        
        # Create video testing loaders.
        if cfg.TEST.OPENSET:
            test_loader = loader.construct_loader(cfg, "test_openset")
        else:
            test_loader = loader.construct_loader(cfg, "test")
        logger.info("Testing model for {} iterations".format(len(test_loader)))

        if cfg.DETECTION.ENABLE:
            assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
            test_meter = AVAMeter(len(test_loader), cfg, mode="test")
        else:
            assert (
                test_loader.dataset.num_videos
                % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
                == 0
            )
            # Create meters for multi-view testing.
            test_meter = TestMeter(
                test_loader.dataset.num_videos
                // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
                cfg.MODEL.NUM_CLASSES
                if not cfg.TASK == "ssl"
                else cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM,
                len(test_loader),
                cfg.DATA.MULTI_LABEL,
                cfg.DATA.ENSEMBLE_METHOD,
            )

        # Set up writer for logging to Tensorboard format.
        if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
            cfg.NUM_GPUS * cfg.NUM_SHARDS
        ):
            writer = tb.TensorboardWriter(cfg)
        else:
            writer = None

        # # Perform multi-view test on the entire dataset.
        test_meter = perform_test(test_loader, model, test_meter, cfg, writer)
        test_meters.append(test_meter)
        if writer is not None:
            writer.close()

    result_string_views = "_p{:.2f}_f{:.2f}".format(params / 1e6, flops)

    for view, test_meter in zip(cfg.TEST.NUM_TEMPORAL_CLIPS, test_meters):
        logger.info(
            "Finalized testing with {} temporal clips and {} spatial crops".format(
                view, cfg.TEST.NUM_SPATIAL_CROPS
            )
        )
        result_string_views += "_{}a{}" "".format(
            view, test_meter.stats["top1_acc"]
        )

        result_string = (
            "_p{:.2f}_f{:.2f}_{}a{} Top5 Acc: {} MEM: {:.2f} f: {:.4f}"
            "".format(
                params / 1e6,
                flops,
                view,
                test_meter.stats["top1_acc"],
                test_meter.stats["top5_acc"],
                misc.gpu_mem_usage(),
                flops,
            )
        )

        logger.info("{}".format(result_string))
    logger.info("{}".format(result_string_views))
    return result_string + " \n " + result_string_views
