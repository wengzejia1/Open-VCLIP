#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import os
from pathlib import Path
import shutil
import submitit
import multiprocessing
import sys
import uuid

import torch
import slowfast.utils.checkpoint as cu
import slowfast.utils.multiprocessing as mpu
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config

from tools.test_net import test
from tools.train_net import train

def parse_args():
    parser = argparse.ArgumentParser(
        "Submitit for onestage training", add_help=False
    )
    parser.add_argument(
        "--num_gpus",
        help="Number of GPUs",
        default=8,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of Nodes",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--partition", 
        default="learnfair", 
        type=str, 
        help="Partition where to submit"
    )
    parser.add_argument(
        "--timeout", 
        default=60 * 48, 
        type=int, 
        help="Duration of the job"
    )
    parser.add_argument(
        "--cfg", 
        dest="cfg_file", 
        help="Path to the config file",
        default="configs/test_R50_8GPU.yaml", type=str
    )
    parser.add_argument(
        "--job_dir", 
        default="/checkpoint/mandelapatrick/slowfast_ssv2", 
        type=str, 
        help="Job dir. Leave empty for automatic."
    )
    parser.add_argument(
        "--name", 
        default="", 
        type=str, 
        help="Job dir. Leave empty for automatic."
    )
    parser.add_argument(
        "--resume-from",
        default="",
        type=str,
        help=(
            "Weights to resume from (.*pth file) or a file (last_checkpoint) that contains "
            + "weight file name from the same directory"
        ),
    )
    parser.add_argument(
        "--resume-job", 
        default="", 
        type=str, 
        help="resume training from the job")
    parser.add_argument(
        "--use_volta32", 
        action='store_true', 
        help="Big models? Use this")
    parser.add_argument(
        "--postfix", 
        default="experiment", 
        type=str, 
        help="Postfix of the jobs"
    )
    parser.add_argument(
        "--mail", 
        default="", 
        type=str,
        help="Email this user when the job finishes if specified"
    )
    parser.add_argument(
        '--comment', 
        default="", 
        type=str,
        help='Comment to pass to scheduler, e.g. priority message'
    )
    parser.add_argument(
        "opts",
        help="See slowfast/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/checkpoint/").is_dir():
        p = Path(f"/checkpoint/{user}/experiments/patching_slowfast")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file():    
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


def launch(shard_id, num_shards, cfg, init_method):
    os.environ["NCCL_MIN_NRINGS"] = "8"

    print ("Pytorch version: ", torch.__version__)
    cfg.SHARD_ID = shard_id
    cfg.NUM_SHARDS = num_shards
    cfg.USE_SBATCH = False

    print([
        shard_id, num_shards, cfg
    ])

    # train, test = get_func(cfg)
    # Launch job.
    if cfg.TRAIN.ENABLE:
        launch_job(cfg=cfg, init_method=init_method, func=train)

    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=init_method, func=test)


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):

        socket_name = os.popen("ip r | grep default | awk '{print $5}'").read().strip('\n')
        print("Setting GLOO and NCCL sockets IFNAME to: {}".format(socket_name))
        os.environ["GLOO_SOCKET_IFNAME"] = socket_name
        os.environ["NCCL_SOCKET_IFNAME"] = socket_name

        hostname_first_node = os.popen(
            "scontrol show hostnames $SLURM_JOB_NODELIST"
        ).read().split("\n")[0]
        dist_url = "tcp://{}:12399".format(hostname_first_node)
        print("We will use the following dist url: {}".format(dist_url))

        self._setup_gpu_args()
        results = launch(
            shard_id=self.args.machine_rank,
            num_shards=self.args.num_shards,
            cfg=load_config(self.args, self.args.cfg_file),
            init_method=dist_url,
        )
        return results

    def checkpoint(self):
        import submitit

        job_env = submitit.JobEnvironment()
        slurm_job_id = job_env.job_id
        if self.args.resume_job == "":
            self.args.resume_job = slurm_job_id
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit

        job_env = submitit.JobEnvironment()
        print(self.args)

        self.args.machine_rank = job_env.global_rank
        self.args.output_dir = str(self.args.output_dir).replace("%j", str(job_env.job_id))
        print(f"Process rank: {job_env.global_rank}")


def main():
    args = parse_args()

    if args.name == "":
        cfg_name = os.path.splitext(os.path.basename(args.cfg_file))[0]
        args.name = '_'.join([cfg_name, args.postfix])

    assert args.job_dir != ""

    args.job_dir = Path(args.job_dir) / "%j"
    args.output_dir = args.job_dir
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    # cluster setup is defined by environment variables
    num_gpus_per_node = args.num_gpus
    nodes = args.num_shards
    partition = args.partition
    timeout_min = args.timeout
    kwargs = {}
    if args.use_volta32:
        kwargs['slurm_constraint'] = 'volta32gb'
    if args.comment:
        kwargs['slurm_comment'] = args.comment

    executor.update_parameters(
        mem_gb=60 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=1,
        cpus_per_task=10 * num_gpus_per_node,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs
    )


    print(args.name)
    executor.update_parameters(name=args.name)

    trainer = Trainer(args)
    job = executor.submit(trainer)
    print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()
