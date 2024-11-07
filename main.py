from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
from models import MockModel
from logger import Logger
import glob
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


probe_train_ds = create_wall_dataloader(
    data_path="/vast/wz1232/dl_final_project_2/probe_normal/train",
    probing=True,
    device=device,
    train=True,
)

probe_val_normal_ds = create_wall_dataloader(
    data_path="/vast/wz1232/dl_final_project_2/probe_normal/val",
    probing=True,
    device=device,
    train=False,
)

probe_val_wall_ds = create_wall_dataloader(
    data_path="/vast/wz1232/dl_final_project_2/probe_wall/val",
    probing=True,
    device=device,
    train=False,
)

probe_val_ds = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}

################################################################################
# TODO: Load your own trained model

model = MockModel()

################################################################################

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
