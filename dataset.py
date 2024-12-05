from typing import NamedTuple, Optional
import torch
import numpy as np
from torchvision import transforms

class WallSample(NamedTuple):
    states: torch.Tensor
    locations: torch.Tensor
    actions: torch.Tensor

class WallDataset:
    def __init__(
        self,
        data_path,
        probing=False,
        device="cuda",
        transform=None,
        vicreg=False,
    ):
        self.device = device
        self.states = np.load(f"{data_path}/states.npy", mmap_mode="r")
        self.actions = np.load(f"{data_path}/actions.npy")

        if probing:
            self.locations = np.load(f"{data_path}/locations.npy")
        else:
            self.locations = None

        self.transform = transform
        self.vicreg = vicreg

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):
        states = torch.from_numpy(self.states[i]).float().to(self.device) # [seq_len, C, H, W]
        actions = torch.from_numpy(self.actions[i]).float().to(self.device)  # [seq_len - 1, action_dim]

        if self.locations is not None:
            locations = torch.from_numpy(self.locations[i]).float()
        else:
            locations = torch.empty(0)

        if self.transform:
            if self.vicreg:
                # Apply transformations to create two views
                states1 = torch.stack([self.transform(state) for state in states])
                states2 = torch.stack([self.transform(state) for state in states])
                return states1, states2, actions
            else:
                # Apply transformation once
                states = torch.stack([self.transform(state) for state in states])
        else:
            pass

        if self.vicreg:
            # For safety, though this case shouldn't occur
            return states, states, actions
        else:
            return WallSample(states=states, locations=locations, actions=actions)

def create_wall_dataloader(
    data_path,
    probing=False,
    device="cuda",
    batch_size=64,
    train=True,
    transform=None,
    vicreg=False,
):
    ds = WallDataset(
        data_path=data_path,
        probing=probing,
        device=device,
        transform=transform,
        vicreg=vicreg,
    )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        drop_last=True,
        pin_memory=False,
    )

    return loader
