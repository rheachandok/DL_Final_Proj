from typing import List, Optional
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from dataset import WallSample
from normalizer import Normalizer
from logger import Logger
from hjepa.models.utils import flatten_conv_output


@torch.no_grad()
def plot_prober_predictions(
    batch: WallSample,
    model: torch.nn.Module,
    prober: torch.nn.Module,
    normalizer: Normalizer,
    name_prefix: str = "",
    idxs: Optional[List[int]] = None,
):
    states = batch.states
    actions = batch.actions

    ################################################################################

    forward_result = model.forward_posterior(
        normalizer.normalize_state(states).transpose(0, 1),
        normalizer.normalize_action(actions).transpose(0, 1),
    )

    pred_encs = forward_result.state_predictions
    pred_encs = flatten_conv_output(pred_encs)
    pred_encs = pred_encs.transpose(0, 1)

    # pred_encs = model(states, actions)

    ################################################################################

    pred_locs = torch.stack([prober(x) for x in pred_encs], dim=0)

    # pred_locs is of shape (batch_size, time, 1, 2)
    if idxs is None:
        idxs = list(range(min(pred_locs.shape[0], 64)))

    gt_locations = batch.locations.cpu()
    pred_locs = normalizer.unnormalize_location(pred_locs).cpu()

    # plot
    for i in tqdm(idxs, desc=f"Plotting {name_prefix}"):  # batch size
        fig = plt.figure(dpi=200)

        images = states

        x_max = images.shape[-2] - 1
        y_max = images.shape[-1] - 1

        plt.imshow(-1 * images[i, 0].sum(dim=0).cpu(), cmap="gray")
        plt.plot(
            gt_locations[i, :, 0].cpu(),
            gt_locations[i, :, 1].cpu(),
            marker="o",
            markersize=2.5,
            linewidth=1,
            c="#3777FF",
            alpha=0.8,
        )
        plt.plot(
            pred_locs[i, :, 0].cpu(),
            pred_locs[i, :, 1].cpu(),
            marker="o",
            markersize=2.5,
            linewidth=1,
            c="#D62828",
            alpha=0.8,
        )
        plt.xlim(0, x_max)
        plt.ylim(y_max, 0)

        Logger.run().log_figure(fig, f"{name_prefix}_prober_predictions_{i}")
        plt.close(fig)
