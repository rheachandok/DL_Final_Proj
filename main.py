from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
from models import MockModel
from logger import Logger
import glob
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# probe_train_ds = create_wall_dataloader(
#     data_path="/vast/wz1232/dl_final_project_2/probe_normal/train",
#     probing=True,
#     device=device,
#     train=True,
# )

# probe_val_normal_ds = create_wall_dataloader(
#     data_path="/vast/wz1232/dl_final_project_2/probe_normal/val",
#     probing=True,
#     device=device,
#     train=False,
# )

# probe_val_wall_ds = create_wall_dataloader(
#     data_path="/vast/wz1232/dl_final_project_2/probe_wall/val",
#     probing=True,
#     device=device,
#     train=False,
# )

# probe_val_ds = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}

probe_train_ds = create_wall_dataloader(
    data_path="/vast/wz1232/dl_final_project_2/probe_expert/train",
    probing=True,
    device=device,
    train=True,
)


probe_val_wall_ds = create_wall_dataloader(
    data_path="/vast/wz1232/dl_final_project_2/probe_expert/val",
    probing=True,
    device=device,
    train=False,
)

probe_val_ds = {"wall_other": probe_val_wall_ds}

################################################################################
# TODO: Load your own trained model


from hjepa.train import TrainConfig
from hjepa.models.hjepa import HJEPA

config = TrainConfig.parse_from_command_line()

model = HJEPA(
    config.hjepa,
    img_size=65,
)

model = model.cuda()

# load ckpt
load_checkpoint_path = (
    "/scratch/wz1232/HJEPA/checkpoint/10-17-1/epoch=100_sample_step=2016768.ckpt"
)
checkpoint = torch.load(load_checkpoint_path)
state_dict = checkpoint["model_state_dict"]
state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

if "backbone.layer1.0.weight" in state_dict:  # this is jepa only model (legacy)
    res = model.level1.load_state_dict(state_dict)
else:
    if config.load_l1_only:
        for k in list(state_dict.keys()):
            # 1. remove all posterior parameters
            # (incompatible because we don't use it in l1).
            # 2. remove everything belonging to l2
            if "level2" in k or "posterior" in k or "decoder.converter" in k:
                del state_dict[k]
            elif "decoder" in k:  # this is for loading RSSM
                del state_dict[k]
    res = model.load_state_dict(state_dict, strict=False)
assert (
    len(res.unexpected_keys) == 0
), f"Unexpected keys when loading weights: {res.unexpected_keys}"
print(f"loaded model from {load_checkpoint_path}")

model = model.level1


# model = MockModel()


################################################################################

remove_all_files("/scratch/wz1232/DL_Final_Proj/media")

Logger.run().initialize(
    output_path="/scratch/wz1232/DL_Final_Proj",
    wandb_enabled=False,
    project="DL_final_project",
    name="test",
    group="test",
)


evaluator = ProbingEvaluator(
    device=device,
    model=model,
    probe_train_ds=probe_train_ds,
    probe_val_ds=probe_val_ds,
    quick_debug=False,
)

prober = evaluator.train_pred_prober()

avg_losses = evaluator.evaluate_all(prober=prober)

for probe_attr, loss in avg_losses.items():
    print(f"{probe_attr} loss: {loss}")
