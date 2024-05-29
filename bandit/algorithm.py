from tqdm import tqdm
import wandb
import torch
from .util import get_model
import pdb
import numpy as np
import math
import random
import copy
from torch.nn import functional as F


class BaseBandit:
    def __init__(self, cfg, env, accelerator):
        self.cfg = cfg
        self.env = env
        self.accelerator = accelerator
        self.model = get_model(cfg, self.env.num_actions)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.cfg.task.lr
        )
        self.model = accelerator.prepare(self.model)
        self.model.to(accelerator.device)
        self.total_regret = 0

        self.history = []

    def run(self):
        for i in range(self.cfg.time_horizon):
            self.accelerator.print("STEP ", i)
            context = self.env.get_context()
            action, stat_dict = self.get_action(i, context)
            cost, optimal_cost = self.env.get_cost(action)
            action = torch.tensor(action).to(self.cfg.device)
            context = self.accelerator.gather(context).cpu()
            action = self.accelerator.gather(action).cpu()
            cost = self.accelerator.gather(cost).cpu()
            optimal_cost = self.accelerator.gather(
                torch.tensor(optimal_cost).to(self.accelerator.device)
            ).cpu()
            self.total_regret += cost.sum() - optimal_cost.sum()
            print("marginal regret: ", cost.sum() - optimal_cost.sum())

            if self.cfg.wandb.use_wandb and self.accelerator.is_local_main_process:
                wandb.log(
                    {
                        "regret": self.total_regret,
                        "cost": cost.mean(),
                        "action": (np.array(action.cpu()) + 1).mean(),
                        "incremental_regret": cost.sum() - optimal_cost.sum(),
                        **stat_dict,
                        # "scale": self.model.scale.item(),
                    }
                )
            if self.accelerator.is_local_main_process:
                print("action ", (np.array(action.cpu())))
                print("cost ", cost.mean())
                print("regret ", self.total_regret)
                print(" ------------------- ")

            if self.cfg.alg == "regcb" and self.cfg.task == "housing":
                cost = cost * 100

            self.history.append((context, action, cost))

        # save model
        torch.save(
            self.accelerator.unwrap_model(self.model).state_dict(), "final_model.pt"
        )

    def risk_for_backprop(self, f):
        total = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.cfg.alg == "distributional":
            for cx, act, cost in (
                random.sample(self.history, self.cfg.task.backprop_batch_size)
                if self.cfg.task.backprop_batch_size < len(self.history)
                else self.history
            ):
                output = f(cx.to(device))
                output = (
                    output.view(self.cfg.task.batch_size,self.env.num_actions,self.cfg.task.num_atoms,)
                )
                
                preds = torch.nn.functional.softmax(output, dim=-1)
                cost_ = []
                for c in cost:
                    if self.cfg.task.task == "housing":
                        cost_.append(int(c * 100))
                    else:
                        cost_.append(self.env.cti[int(c * 100)])
                # pdb.set_trace()
                total += torch.sum(
                    -1
                    * torch.log(
                        preds[
                            range(self.cfg.task.batch_size),
                            act,
                            cost_,
                        ]
                    ),
                )

            total /= (
                (
                    self.cfg.task.backprop_batch_size
                    if self.cfg.task.backprop_batch_size < len(self.history)
                    else len(self.history)
                )
                * self.cfg.task.batch_size
                * self.accelerator.num_processes
            )
        else:
            for cx, act, cost in (
                random.sample(self.history, self.cfg.task.backprop_batch_size)
                if self.cfg.task.backprop_batch_size < len(self.history)
                else self.history
            ):
                total += F.mse_loss(f(cx.to("cuda")).cpu()[range(len(act)), act], cost)

            total /= (
                self.cfg.task.backprop_batch_size
                if self.cfg.task.backprop_batch_size < len(self.history)
                else len(self.history)
            )
        return total

    @torch.no_grad()
    def risk(self, f):
        total = 0
        if self.cfg.alg == "distributional":
            i = 0
            for cx, act, cost in self.history:
                output = f(cx).view(
                    self.cfg.task.batch_size,
                    self.env.num_actions,
                    self.cfg.task.num_atoms,
                )
                preds = torch.nn.functional.softmax(output, dim=-1)
                cost_ = []
                for c in cost:
                    if self.cfg.task.task == "housing":
                        cost_.append(int(c * 100))
                    else:
                        cost_.append(self.env.cti[int(c * 100)])

                total += torch.sum(
                    -1
                    * torch.log(
                        preds[
                            range(self.cfg.task.batch_size),
                            act,
                            cost_,
                        ]
                    )
                )
                if i == 0:
                    print(
                        torch.mean(
                            preds[
                                range(self.cfg.task.batch_size),
                                act,
                                cost_,
                            ]
                        )
                    )
                print("total is ", total)
                i += 1
        else:
            for cx, act, cost in self.history:
                total += (
                    (f(cx.to("cuda")).cpu()[range(len(act)), act] - cost) ** 2
                ).sum()
                cx.to("cpu")
        total /= (
            len(self.history)
            * self.cfg.task.batch_size
            * self.accelerator.num_processes
        )

        return total

    def get_mse(self, f):
        total = 0
        for cx, act, cost in self.history:
            output = f(cx).view(
                self.cfg.task.batch_size,
                self.env.num_actions,
                self.cfg.task.num_atoms,
            )
            preds = torch.nn.functional.softmax(output, dim=-1)
            expected_values = torch.einsum("ijk,k->ij", preds, self.env.ev_mapping)
            total += ((expected_values[range(len(act)), act] - cost) ** 2).sum()
        return total / (
            len(self.history)
            * self.cfg.task.batch_size
            * self.accelerator.num_processes
        )

    def width_objective(self, f, f_prime, d_q, d_j, lambda_=1e-4, lambda_1=1e-9):
        total = 0

        total -= lambda_ * ((f(d_q) - f_prime(d_q)) ** 2).cpu().sum() / len(d_q)

        sub_total = 0
        for cx, act, _ in d_j:
            sub_total += (
                (
                    f(cx.to("cuda")).cpu()[range(len(act)), act]
                    - f_prime(cx.to("cuda")).cpu()[range(len(act)), act]
                )
                ** 2
            ).sum()

        total += sub_total / (
            len(d_j) * self.cfg.task.batch_size * self.accelerator.num_processes
        )
        print(sub_total / (len(d_j) * self.cfg.task.batch_size))

        total += lambda_1 * (f_prime(d_q) - f(d_q)).cpu().sum() / len(d_q)

        return total

    def compute_expected_costs(self, f, context):
        output = f(context)
        output = output.view(
            self.cfg.task.batch_size,
            self.env.num_actions,
            self.cfg.task.num_atoms,
        )
    
        preds = torch.nn.functional.softmax(output, dim=-1)
        expected_values = torch.einsum("ijk,k->ij", preds, self.env.ev_mapping)
        return expected_values

    def distributional_width_objective(
        self, f, f_prime, d_q, d_j, lambda_=1e-4, lambda_1=1e-9
    ):
        total = 0

        total -= (
            lambda_
            * (
                (
                    self.compute_expected_costs(f, d_q)
                    - self.compute_expected_costs(f_prime, d_q)
                )
                ** 2
            )
            .cpu()
            .sum()
            / len(d_q)
        )

        sub_total = 0
        for cx, act, _ in d_j:
            sub_total += (
                (
                    self.compute_expected_costs(f, cx.to("cuda")).cpu()[
                        range(len(act)), act
                    ]
                    - self.compute_expected_costs(f_prime, cx.to("cuda")).cpu()[
                        range(len(act)), act
                    ]
                )
                ** 2
            ).sum()

        total += sub_total / (
            len(d_j) * self.cfg.task.batch_size * self.accelerator.num_processes
        )

        total += (
            lambda_1
            * (
                self.compute_expected_costs(f_prime, d_q)
                - self.compute_expected_costs(f, d_q)
            )
            .cpu()
            .sum()
            / len(d_q)
        )

        return total

    def calculate_bonuses(self, context, step):
        new_model = get_model(self.cfg, self.env.num_actions)
        new_model_optimizer = torch.optim.Adam(
            new_model.parameters(),
            lr=1e-4,
        )

        new_model = new_model.to(self.accelerator.device)
        new_model.load_state_dict(
            self.accelerator.unwrap_model(self.model).state_dict()
        )

        for _ in range(5):
            # sample minibatch from context
            if len(self.history) < self.cfg.task.backprop_batch_size:
                minibatch_j = self.history
            else:
                minibatch_j = random.sample(
                    self.history, self.cfg.task.backprop_batch_size
                )

            # calculate gradient
            new_model_optimizer.zero_grad()
            if self.cfg.alg == "distributional":
                loss = self.distributional_width_objective(
                    self.model,
                    new_model,
                    context,
                    minibatch_j,
                )
            else:
                loss = self.width_objective(
                    self.model,
                    new_model,
                    context,
                    minibatch_j,
                )
            print("LOSS: ", loss.item())
            self.accelerator.backward(loss)
            # clip grad norm
            torch.nn.utils.clip_grad_norm_(new_model.parameters(), 5)
            new_model_optimizer.step()
        # calculate bonuses
        if self.cfg.alg == "distributional":
            width = abs(
                self.compute_expected_costs(self.model, context)
                - self.compute_expected_costs(new_model, context)
            )
        else:
            width = abs(self.model(context) - new_model(context))
        mean_width = width.mean()
        std_width = width.std(dim=-1).mean()
        print("WIDTH: ", width.mean())
        max_width = width.max()
        annealing_rate = 1
        coeff = 0.5
        return (
            annealing_rate * coeff * (width / max_width),
            mean_width,
            std_width,
            annealing_rate,
        )

    def get_action(self, step, context):
        # warmup stage
        if step < self.cfg.task.warmup_steps:
            return (
                torch.randint(self.env.num_actions, (context.shape[0],)).tolist(),
                {},
            )
        else:
            # get action
            print("Optimizing")

            if self.cfg.task == "prudential":
                target = (
                    31 / 32
                    if self.cfg.alg == "distributional"
                    else 0.07  # 0.0
                )
            elif self.cfg.task == "housing":
                target = 7 if self.cfg.alg == "regcb" else 0.1
            else:
                target = 1 / 32 if self.cfg.alg == "distributional" else 0.01
            itr = 0

            while (loss := self.risk_for_backprop(self.model)) > target or itr < 5:
                if torch.isinf(loss):
                    pdb.set_trace()
                print("RISK ", loss)

                self.optimizer.zero_grad()
                self.accelerator.backward(loss)

                # clip grad norm
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

                self.optimizer.step()
                itr += 1
                if itr > 100:
                    break

            stat_dict = {
                "critic_loss": loss.item(),
            }
            if not self.cfg.is_optimistic:
                if self.cfg.alg == "regcb":
                    return self.model(context).argmin(dim=-1).tolist(), stat_dict
                else:
                    model_expected_costs = self.compute_expected_costs(self.model, context)
                    actions = model_expected_costs.argmin(dim=-1).tolist()
                    return actions, stat_dict
            bonuses, mean_width, std_width, annealing_rate = self.calculate_bonuses(
                context, step
            )

            stat_dict["mean_width"] = mean_width
            stat_dict["std_width"] = std_width
            stat_dict["annealing_rate"] = annealing_rate

            print("CALCULATING BONUSES")

            if self.cfg.alg == "distributional":
                model_expected_costs = self.compute_expected_costs(self.model, context)
            else:
                model_expected_costs = self.model(context)

            mean_deviation_from_greedy = (
                (
                    (model_expected_costs - bonuses).argmin(dim=-1)
                    == model_expected_costs.argmin(dim=-1)
                )
                .to(torch.float)
                .mean()
            )

            stat_dict["deviation_from_greedy"] = mean_deviation_from_greedy
            stat_dict["bonuses"] = bonuses.flatten().tolist()

            if self.cfg.alg == "regcb":
                return (
                    (self.model(context) - bonuses).argmin(dim=-1).tolist(),
                    stat_dict,
                )
            else:
                best_actions = (model_expected_costs - bonuses).argmin(dim=-1).tolist()

                return best_actions, stat_dict
