from dataset import WallDataset, create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
from models import MockModel
import glob
from impl import JEPA, train_model
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
import random
#from torch.utils.tensorboard import SummaryWrite

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
    #model = MockModel()
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


class RandomResizedCropTensor:
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3./4., 4./3.)):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img):
        # img: Tensor of shape (C, H, W)
        i, j, h, w = transforms.RandomResizedCrop.get_params(img, self.scale, self.ratio)
        img = F.resized_crop(img, i, j, h, w, self.size)
        return img

class RandomHorizontalFlipTensor:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        # img: Tensor of shape (C, H, W)
        if random.random() < self.p:
            img = torch.flip(img, dims=[2])  # Flip width dimension
        return img

class RandomVerticalFlipTensor:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        # img: Tensor of shape (C, H, W)
        if random.random() < self.p:
            img = torch.flip(img, dims=[1])  # Flip height dimension
        return img


class AddGaussianNoise:
    def __init__(self, mean=0., std=0.02):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        noise = torch.randn_like(img) * self.std + self.mean
        img = img + noise
        return img.clamp(0, 1)
if __name__ == "__main__":

    device = get_device()
    
    # Load training and validation datasets for probing (if needed)
    probe_train_ds, probe_val_ds = load_data(device)
    
    # Initialize the model
    model = load_model()
    model.apply(weights_init)


    vicreg_transforms = transforms.Compose([
        RandomResizedCropTensor(size=(65, 65), scale=(0.8, 1.0)),
        RandomHorizontalFlipTensor(p=0.5),
        RandomVerticalFlipTensor(p=0.5),
        AddGaussianNoise(mean=0., std=0.02),
       # Add more custom transforms if needed
    ])

    # Create data loaders
    train_loader = create_wall_dataloader(
        data_path='/scratch/DL24FA/train',
        probing=False,
        device=device,
        train=True,
        transform=vicreg_transforms,
        vicreg=True,
    )

   
    # Initialize TensorBoard writer (optional)
    #writer = SummaryWriter(log_dir='runs/vicreg_experiment')
    
    # Create the optimizer
    optimizer = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.predictor.parameters()),
        lr=1e-4
    )
    
    # Train the model
    train_model(model, train_loader, optimizer, num_epochs=10, device=device)
    
    # Evaluate the model
    evaluate_model(device, model, probe_train_ds, probe_val_ds)
