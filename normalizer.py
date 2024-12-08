import torch


class Normalizer:
    def __init__(self):
        self.location_mean = torch.tensor([31.5863, 32.0618])
        self.location_std = torch.tensor([16.1025, 16.1353])

    def normalize_embedding(self, location: torch.Tensor) -> torch.Tensor:
        return (location - self.location_mean.to(location.device)) / (
            self.location_std.to(location.device) + 1e-6
        )
