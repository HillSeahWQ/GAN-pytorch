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
            scalars = {
                "loss_disc_real": loss_real_D.item(),
                "loss_disc_fake": loss_fake_D.item(),
                "loss_disc": loss_D.item(),
                "loss_gen": loss_G.item()
            }
            logger.add_scalars("Losses", scalars, global_step=epoch * l + batch_idx)

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

                        logger.add_image("Real", img_grid_real, global_step=epoch * l + batch_idx)
                        logger.add_image("Fake", img_grid_fake, global_step=epoch * l + batch_idx)
                step += 1

        # save models' checkpoint after each epoch
        torch.save(generator.state_dict(), os.path.join("models", args.run_name, f"gen_ckpt.pt"))
        torch.save(discriminator.state_dict(), os.path.join("models", args.run_name, f"disc_ckpt.pt"))
        torch.save(optimizer_G.state_dict(), os.path.join("models", args.run_name, f"gen_optim.pt"))
        torch.save(optimizer_D.state_dict(), os.path.join("models", args.run_name, f"disc_optim.pt"))        
    

def launch():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # setup
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.dataset_path = Path().cwd().parent / "data" / "celeb_A" # Path().cwd().parent / "data" / "celeb_A"
    args.run_name = "DCGAN"
    args.epochs = 10
    # Hyperparameters following the DCGAN paper
    args.transforms=None # follow default transforms
    args.batch_size = 128
    args.img_size = 64
    args.img_channels = 3 # can be changed wrt to images (althought DCGAN paper, input channels of images is to be 3)
    args.z_dim = 100
    args.lr = 2e-4
    args.b1 = 0.5
    train(args)


if __name__ == "__main__":
     launch()