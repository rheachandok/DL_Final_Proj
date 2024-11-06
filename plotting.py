from typing import List, Optional
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from dataset import WallSample
from hjepa.models.jepa import JEPA
from hjepa.data.normalizer import Normalizer


@torch.no_grad()
def plot_prober_predictions(
    batch: WallSample,
    jepa: JEPA,
    prober: torch.nn.Module,
    normalizer: Normalizer,
    name_prefix: str = "",
    idxs: Optional[List[int]] = None,
    notebook: bool = False,
    pixel_mapper=None,
):
    # infer
    states = batch.states.cuda().transpose(0, 1)

    # drop actions of other spheres, put time first
    actions = batch.actions[:, :, 0].cuda().transpose(0, 1)

    if hasattr(batch, "propio_states"):
        ps = batch.propio_states.cuda().transpose(0, 1)
    else:
        ps = None

    forward_result = jepa.forward_posterior(states, actions, propio_states=ps)
    pred_encs = forward_result.state_predictions

    pred_locs = torch.stack([prober(x) for x in pred_encs], dim=1)
    # pred_locs is of shape (batch_size, time, 1, 2)
    if idxs is None:
        idxs = list(range(min(pred_locs.shape[0], 64)))

    gt_locations = normalizer.unnormalize_location(batch.locations).cpu()
    pred_locs = normalizer.unnormalize_location(pred_locs).cpu()

    if pixel_mapper is not None:
        gt_locations = pixel_mapper(gt_locations)
        pred_locs = pixel_mapper(pred_locs)

    # plot
    for i in tqdm(idxs, desc=f"Plotting {name_prefix}"):  # batch size
        fig = plt.figure(dpi=200)
        if hasattr(batch, "view_states") and batch.view_states.shape[-1]:
            images = batch.view_states
        else:
            images = batch.states

        x_max = images.shape[-2] - 1
        y_max = images.shape[-1] - 1

        plt.imshow(-1 * images[i, 0].sum(dim=0).cpu(), cmap="gray")
        plt.plot(
            gt_locations[i, :, 0, 0].cpu(),
            gt_locations[i, :, 0, 1].cpu(),
            marker="o",
            markersize=2.5,
            linewidth=1,
            c="#3777FF",
            alpha=0.8,
        )
        plt.plot(
            pred_locs[i, :, 0, 0].cpu(),
            pred_locs[i, :, 0, 1].cpu(),
            marker="o",
            markersize=2.5,
            linewidth=1,
            c="#D62828",
            alpha=0.8,
        )
        plt.xlim(0, x_max)
        plt.ylim(y_max, 0)

        if not notebook:
            Logger.run().log_figure(fig, f"{name_prefix}/prober_predictions_{i}")
            plt.close(fig)
