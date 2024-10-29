# utils.py
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision


def initialize_weights(model): # Weights are initialized from Normal Distribution with mean = 0; standard deviation = 0.02.
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def get_dataloader(img_dir, batch_size=64, img_channels=3, img_size=64, transforms=None):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(size=(img_size, img_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.5 for _ in range(img_channels)],
                [0.5 for _ in range(img_channels)]
            )
        ]
    )

    dataset = torchvision.datasets.ImageFolder(root=img_dir, transform=transforms)

    # Create the dataloader
    NUM_WORKERS = os.cpu_count()

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        shuffle=True,
        pin_memory=True
    )

    return dataloader


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)