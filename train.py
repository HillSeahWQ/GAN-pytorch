# train.py
import os
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
from utils import setup_logging, get_dataloader, initialize_weights
from modules import Generator, Discriminator

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def train(args):
    # Setup
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_dataloader(args.dataset_path, args.batch_size, args.img_channels, args.img_size, args.transforms)
    generator = Generator(noise_channels=args.z_dim, img_channels=args.img_channels).to(device)
    discriminator = Discriminator(img_channels=args.img_channels).to(device)
    criterion = nn.BCELoss()
    optimizer_D = optim.Adam(params=discriminator.parameters(), lr=args.lr, betas=(args.b1, 0.999)) # b2 kept as default
    optimizer_G = optim.Adam(params=generator.parameters(), lr=args.lr, betas=(args.b1, 0.999))  # b2 kept as default
    fixed_noise = torch.randn(args.batch_size, args.z_dim, 1, 1).to(device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    initialize_weights(generator)
    initialize_weights(discriminator)
    step = 0

    for epoch in tqdm(range(args.epochs)):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)

        for batch_idx, (x, _) in enumerate(pbar):
            x = x.to(device)
            noise = torch.randn(size = (x.shape[0], args.z_dim, 1, 1)).to(device)

            g_z = generator(noise) # G(z)
            d_x = discriminator(x).reshape(-1) # D(x), reshape from 1*1*1 to 1
            d_g_z = discriminator(g_z).reshape(-1) # D(G(z)), reshape from 1*1*1 to 1

            ### Train the Discriminator: Min -(log(D(x)) + log(1-D(G(Z)))) <---> Max log(D(x)) + log(1-D(G(Z)))
            discriminator.train()
            generator.eval()

            loss_real_D = criterion(d_x, torch.ones_like(d_x)) # -log(D(X))
            loss_fake_D = criterion(d_g_z, torch.zeros_like(d_g_z)) # -log(1-D(G(z)))
            loss_D = (loss_fake_D + loss_real_D )/ 2 # -(log(D(x)) + log(1-D(G(Z))))

            optimizer_D.zero_grad()

            loss_D.backward(retain_graph=True)

            optimizer_D.step()

            ### Train the Generator: Min -log(D(G(z)) <---> Max log(D(G(z))) <---> Min log(1-D(G(z)))
            generator.train()
            discriminator.eval()

            d_g_z_next = discriminator(g_z).reshape(-1) # after training the disc, new D(G(z)), reshape from 1*1*1 to 1
            loss_G = criterion(d_g_z_next, torch.ones_like(d_g_z_next)) # -log(D(G(z)))

            optimizer_G.zero_grad()

            loss_G.backward()

            optimizer_G.step()

            # Logs
            pbar.set_description(f"Epoch [{epoch} / {args.epochs}]")
            pbar.set_postfix(loss_disc = loss_D.item(), loss_gen = loss_G.item())

            # Evaluation
            if batch_idx % 50 == 0:
                scalars = {
                    "loss_disc_real": loss_real_D.item(),
                    "loss_disc_fake": loss_fake_D.item(),
                    "loss_disc": loss_D.item(),
                    "loss_gen": loss_G.item()
                }
                logger.add_scalars("Losses", scalars, global_step=step)
                with torch.no_grad():
                        generator.eval()
                        fake = generator(fixed_noise)
                        # take out (up to) 32 examples
                        img_grid_real = torchvision.utils.make_grid(x[:32], normalize=True)
                        img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                        logger.add_image("Real", img_grid_real, global_step=step)
                        logger.add_image("Fake", img_grid_fake, global_step=step)
                step += 1

        # save models' checkpoint after each epoch
        torch.save(generator.state_dict(), os.path.join("models", args.run_name, f"gen_ckpt.pt"))
        torch.save(discriminator.state_dict(), os.path.join("models", args.run_name, f"disc_ckpt.pt"))
        torch.save(optimizer_G.state_dict(), os.path.join("models", args.run_name, f"gen_optim.pt"))
        torch.save(optimizer_D.state_dict(), os.path.join("models", args.run_name, f"disc_optim.pt"))        
    

def launch():
    parser = argparse.ArgumentParser(description="Train a DCGAN on a dataset.")

    # setup
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run the training on (default: cuda if available, else cpu).")
    parser.add_argument("--dataset_path", type=str, default=r"D:\pompous_penguin\data\celeb_A", 
                        help="Path to the dataset (default: D:\\pompous_penguin\\data\\celeb_A).")
    parser.add_argument("--run_name", type=str, default="DCGAN", 
                        help="Name of the run (default: DCGAN).")
    parser.add_argument("--epochs", type=int, default=500, 
                        help="Number of epochs to train for (default: 500).")
    # Hyperparameters following the DCGAN paper
    parser.add_argument("--batch_size", type=int, default=128, 
                        help="Batch size for training (default: 128).")
    parser.add_argument("--img_size", type=int, default=64, 
                        help="Size of the images (default: 64).")
    parser.add_argument("--img_channels", type=int, default=3, # can be changed wrt to images (althought DCGAN paper, input channels of images is to be 3)
                        help="Number of channels in the images (default: 3).")
    parser.add_argument("--z_dim", type=int, default=100, 
                        help="Dimension of the noise vector (default: 100).")
    parser.add_argument("--lr", type=float, default=2e-4, 
                        help="Learning rate for the optimizers (default: 2e-4).")
    parser.add_argument("--b1", type=float, default=0.5, 
                        help="Beta1 hyperparameter for Adam optimizer (default: 0.5).")
    # Parse arguments
    args = parser.parse_args()

    # Setup additional attributes if needed (e.g., transforms)
    args.transforms = None  # Follow default transforms

    # Train the model
    train(args)


if __name__ == "__main__":
     launch()