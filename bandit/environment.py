from .dataloader import get_dataloader
import pdb
from .util import (
    fine_to_coarse,
    prudential_cost_to_idx,
    cifar_cost_to_idx,
)
import numpy as np
import torch


class BaseEnvironment:
    def __init__(self, cfg, accel):
        self.cfg = cfg
        self.accel = accel
        self.dataloader, self.num_actions = get_dataloader(cfg, self.accel)

        self.curr_labels = None

        if cfg.task.task == "cifar100":
            self.ftc = np.vectorize(fine_to_coarse.get)
            self.cti = cifar_cost_to_idx
            self.ev_mapping = torch.tensor([0, 0.5, 1.0]).to("cuda")
        elif cfg.task.task == "prudential":
            self.cti = prudential_cost_to_idx
            self.ev_mapping = torch.tensor(
                [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1]
            ).to("cuda")
        elif cfg.task.task == "housing":
            self.ev_mapping = (torch.arange(0, 1.01, 0.01)).to("cuda")

    def get_context(self):
        _, (ctx, label) = next(enumerate(self.dataloader))
        ctx = ctx.to(self.cfg.device)
        label = label.to(self.cfg.device)
        self.curr_labels = label
        return ctx

    def get_true_label(self):
        return self.curr_labels

    def get_cost(self, action):
        if self.cfg.task.task == "cifar100":
            print("TRUE LABELS", self.curr_labels)
            action = np.array(action)
            # see which coarse label was chosen
            coarse_prediction = torch.tensor(self.ftc(action)).to(self.cfg.device)

            coarse_true = torch.tensor(
                self.ftc(self.curr_labels.detach().cpu().numpy())
            ).to(self.cfg.device)

            # compute cost
            # 1 if wrong coarse label and wrong fine label
            # 0.5 if right coarse label but wrong fine label
            # 0 if right coarse label and right fine label

            cost = torch.where(
                coarse_prediction == coarse_true,
                torch.where(
                    torch.tensor(action).to(self.cfg.device) == self.curr_labels, 0, 0.5
                ),
                1,
            )

            return cost, 0

        elif self.cfg.task.task == "prudential":
            action = torch.tensor(action).to(self.cfg.device) + 1
            over_predict = action > self.curr_labels
            cost = torch.where(
                over_predict,
                1.0,
                0.1 * (self.curr_labels - action).to(dtype=torch.int),
            )
            return cost, 0
        elif self.cfg.task.task == "housing":
            action = torch.tensor(action).to(self.cfg.device) / 100
            over_predict = action > self.curr_labels.reshape(-1)
            cost = torch.where(over_predict, 1.0, 1.0 - action)
            return cost, (1 - self.curr_labels.reshape(-1)).sum()
        else:
            raise NotImplementedError(f"Task {self.cfg.task.task} not implemented")
