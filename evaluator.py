from typing import NamedTuple, List, Any, Optional, Dict
from itertools import chain
from dataclasses import dataclass
import itertools
import os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
from matplotlib import pyplot as plt

from schedulers import Scheduler, LRSchedule
from models import Prober, build_mlp
from plotting import plot_prober_predictions
from configs import ConfigBase

from dataset import WallDataset

from hjepa.data.enums import ProbingDatasets, DatasetType


@dataclass
class ProbingConfig(ConfigBase):
    probe_targets: str = "locations"
    lr: float = 0.0002
    epochs: int = 20
    schedule: LRSchedule = LRSchedule.Constant
    sample_timesteps: int = 30
    prober_arch: str = "512"
    visualize_probing: bool = True


class ProbeResult(NamedTuple):
    model: torch.nn.Module
    average_eval_loss: float
    eval_losses_per_step: List[float]
    plots: List[Any]


default_config = ProbingConfig()


def location_losses(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert pred.shape == target.shape
    # Pred and target are both B x T x N_DOTS x 2 or B x N_DOTS x 2.
    # we just avg the batch.
    # mse = (pred - target).pow(2).flatten(end_dim=-4).mean(dim=0)
    mse = (pred - target).pow(2).mean(dim=0)
    return mse


class ProbingEvaluator:
    def __init__(
        self,
        device: "cuda",
        model: torch.nn.Module,
        probe_train_ds,
        probe_val_ds: dict,
        config: ProbingConfig = default_config,
        quick_debug: bool = False,
    ):
        self.device = device
        self.config = config
        self.model = model
        self.quick_debug = quick_debug

        self.ds = probe_train_ds
        self.val_ds = probe_val_ds

    def train_pred_prober(self):
        """
        Probes whether the predicted embeddings capture the future locations
        """

        repr_dim = self.model.output_dim
        dataset = self.ds
        model = self.model

        config = self.config
        epochs = config.epochs

        if self.quick_debug:
            epochs = 1
        test_batch = next(iter(dataset))

        prober_output_shape = getattr(test_batch, "locations")[0, 0].shape
        prober = Prober(
            repr_dim,
            config.prober_arch,
            output_shape=prober_output_shape,
        ).to(self.device)

        all_parameters = []
        all_parameters += list(prober.parameters())

        optimizer_pred_prober = torch.optim.Adam(all_parameters, config.lr)

        step = 0

        batch_size = dataset.batch_size
        batch_steps = None

        scheduler = Scheduler(
            schedule=self.config.schedule,
            base_lr=config.lr,
            data_loader=dataset,
            epochs=epochs,
            optimizer=optimizer_pred_prober,
            batch_steps=batch_steps,
            batch_size=batch_size,
        )

        for epoch in tqdm(range(epochs), desc=f"Probe prediction epochs"):
            for batch in tqdm(dataset, desc="Probe prediction step"):
                pred_encs = model(states=batch.states, actions=batch.actions)
                pred_encs = pred_encs.detach()

                # BS, T, D --> T, BS, D
                pred_encs = pred_encs.transpose(0, 1)

                n_steps = pred_encs.shape[0]
                bs = pred_encs.shape[1]

                losses_list = []

                target = getattr(batch, "locations").cuda()

                if (
                    config.sample_timesteps is not None
                    and config.sample_timesteps < n_steps
                ):
                    sample_shape = (config.sample_timesteps,) + pred_encs.shape[1:]
                    # we only randomly sample n timesteps to train prober.
                    # we most likely do this to avoid OOM
                    sampled_pred_encs = torch.empty(
                        sample_shape,
                        dtype=pred_encs.dtype,
                        device=pred_encs.device,
                    )

                    sampled_target_locs = torch.empty(bs, config.sample_timesteps, 1, 2)

                    for i in range(bs):
                        indices = torch.randperm(n_steps)[: config.sample_timesteps]
                        sampled_pred_encs[:, i, :] = pred_encs[indices, i, :]
                        sampled_target_locs[i, :] = target[i, indices]

                    pred_encs = sampled_pred_encs
                    target = sampled_target_locs.cuda()

                pred_locs = torch.stack([prober(x) for x in pred_encs], dim=1)
                losses = location_losses(pred_locs, target)
                per_probe_loss = losses.mean()

                if step % 100 == 0:
                    print(f"finetune_pred_locations loss {per_probe_loss.item()}")

                losses_list.append(per_probe_loss)
                optimizer_pred_prober.zero_grad()
                loss = sum(losses_list)
                loss.backward()
                optimizer_pred_prober.step()

                lr = scheduler.adjust_learning_rate(step)

                step += 1

        return prober

    @torch.no_grad()
    def evaluate_all(
        self,
        probers,
        epoch,
        pixel_mapper=None,
        visualize=True,
    ):
        """
        Evaluates on all the different validation datasets
        """

        val_datasets = {"pred_probe": self.val_ds}
        val_datasets.update(self.extra_val_ds)

        for prefix, val_ds in val_datasets.items():
            self.evaluate_pred_prober(
                probers=probers,
                epoch=epoch,
                val_ds=val_ds,
                pixel_mapper=pixel_mapper,
                visualize=visualize,
            )

    @torch.no_grad()
    def evaluate_pred_prober(
        self,
        probers,
        epoch,
        val_ds: DatasetType,
        pixel_mapper=None,
        visualize=True,
    ):
        quick_debug = self.quick_debug
        config = self.config

        eval_repr_losses = []

        probing_losses = {}
        for probe_target, prober in probers.items():
            prober.eval()
            probing_losses[probe_target] = []

        for idx, batch in enumerate(tqdm(val_ds, desc="Eval probe pred")):
            # put time first
            states = batch.states.cuda().transpose(0, 1)

            # drop actions of other spheres, put time first
            actions = batch.actions[:, :, 0].cuda().transpose(0, 1)

            forward_result = model.forward_posterior(states, actions)

            pred_encs = forward_result.state_predictions
            encs = forward_result.encodings

            for probe_target, prober in probers.items():
                target = getattr(batch, probe_target).cuda()
                pred_locs = torch.stack([prober(x) for x in pred_encs], dim=1)
                losses = location_losses(pred_locs, target)
                probing_losses[probe_target].append(losses.cpu())

            repr_loss = F.mse_loss(encs, pred_encs)
            eval_repr_losses.append(repr_loss.cpu())

            if quick_debug and idx > 2:
                break

        repr_loss = torch.stack(eval_repr_losses).mean()

        log_dict = {
            f"finetune_pred_val/repr_loss": repr_loss.item(),
        }

        for probe_target, eval_losses in probing_losses.items():
            losses_t = torch.stack(eval_losses, dim=0).mean(dim=0)
            losses_t = val_ds.normalizer.unnormalize_mse(losses_t, probe_target)
            losses_t = losses_t.mean(dim=-1)
            average_eval_loss = losses_t.mean().item()
            log_dict[f"finetune_pred_val_{probe_target}/loss"] = average_eval_loss
            log_dict[f"finetune_pred_val_{probe_target}/loss_rmse"] = np.sqrt(
                average_eval_loss
            )

            time_skip = 3  # for less logging

            for i, val_loss in enumerate(losses_t[::time_skip]):
                actual_i = i * time_skip
                log_dict[
                    f"finetune_pred_val_{probe_target}/loss_{actual_i}"
                ] = val_loss.item()

        # right now, we only visualize location predictions
        if self.config.visualize_probing and visualize:
            plot_prober_predictions(
                next(iter(val_ds)),
                model,
                probers["locations"],
                normalizer=val_ds.normalizer,
                name_prefix="",
                idxs=None if not quick_debug else list(range(10)),
                pixel_mapper=pixel_mapper,
            )

        return
