from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
from models import MockModel
import glob
from impl import JEPA, train_model

def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_data(device):
    data_path = "/scratch/DL24FA"

    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_ds = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}
    return probe_train_ds, probe_val_ds


def load_model():
    """Load or initialize the model."""
    # TODO: Replace MockModel with your trained model
    device = get_device()
    model = JEPA(input_channels=2, hidden_dim=256, action_dim=2).to(device)
    return model


def evaluate_model(device, model, probe_train_ds, probe_val_ds):
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


def weights_init(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


if __name__ == "__main__":
    device = get_device()
    probe_train_ds, probe_val_ds = load_data(device)
    #for batch in probe_train_ds:
        #states, locations, actions = batch
        #print(states.shape)
        #print(locations.shape)
        #print(actions.shape)
    model = load_model()
    optimizer = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.predictor.parameters()),
        lr=1e-4
    )
    training = create_wall_dataloader(
        data_path=f"/scratch/DL24FA/train",
        probing=False,
        device=device,
        train=True,
    )
    model.apply(weights_init)
    train_model(model, training, optimizer, num_epochs=10, device=device)
    evaluate_model(device, model, probe_train_ds, probe_val_ds)
