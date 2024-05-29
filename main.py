import hydra
from omegaconf import DictConfig
from bandit.dataloader import get_dataloader
from bandit.algorithm import BaseBandit
from bandit.environment import BaseEnvironment
import wandb
from accelerate import Accelerator
import pdb
import torch
import numpy as np
from accelerate.utils import DistributedDataParallelKwargs
import random

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # set seed
    if cfg.seed is not None:
        print("Setting seed to", cfg.seed)
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    kwargs = DistributedDataParallelKwargs(broadcast_buffers=False)
    accel = Accelerator(kwargs_handlers=[kwargs])

    if cfg.wandb.use_wandb and accel.is_local_main_process:
        name = f"seed={cfg.seed},alg={cfg.alg},task={cfg.task.task}" 
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=name,
            config=dict(cfg),
        )

    enviroment = BaseEnvironment(cfg, accel)
    bandit = BaseBandit(cfg, enviroment, accel)
    bandit.run()


if __name__ == "__main__":
    main()
