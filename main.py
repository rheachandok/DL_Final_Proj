from dataset import WallDataset
from evaluator import ProbingEvaluator
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

probe_train_ds = WallDataset(
    data_path="/vast/wz1232/dl_final_project_2/probe_normal/train",
    probing=True,
    device=device,
)

probe_val_normal_ds = WallDataset(
    data_path="/vast/wz1232/dl_final_project_2/probe_normal/val",
    probing=True,
    device=device,
)

probe_val_wall_ds = WallDataset(
    data_path="/vast/wz1232/dl_final_project_2/probe_wall/val",
    probing=True,
    device=device,
)

probe_val_ds = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}

evaluator = ProbingEvaluator(
    device=device, model=None, probe_train_ds=probe_train_ds, probe_val_ds=probe_val_ds
)
