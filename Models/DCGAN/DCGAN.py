import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Models.DCGAN.Discriminator import Discriminator
from Models.DCGAN.Generator import Generator

# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 2e-4  # could also use two lrs, one for gen and one for disc
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 1
NOISE_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64


def initialize_weights(layer):
    if type(layer) == nn.Conv2d or type(layer) == nn.ConvTranspose2d:
        layer.weight.data.normal_(0, .2)

'''
Deep convolutional GAN
'''


class DCGAN:
    def __init__(self, num_epochs, verbose, save_path):
        self.num_epochs = num_epochs
        self.verbose = verbose  # 0 - print nothing, 1 print batch statistics every 100 batches
        self.save_path = save_path
        self.data_transforms = transforms.Compose(
            [
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
                ),
            ])
        # If you train on MNIST, remember to set channels_img to 1
        self.dataset = datasets.MNIST(root="dataset/", train=True, transform=self.data_transforms,
                                      download=True)

    def train(self):
        # comment mnist above and uncomment below if train on CelebA
        # dataset = datasets.ImageFolder(root="celeb_dataset", transform=transforms)
        dataloader = DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True)

        gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
        disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)

        gen.apply(initialize_weights)
        disc.apply(initialize_weights)

        opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
        opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
        loss = nn.BCELoss()

        fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
        writer_real = SummaryWriter(f"{self.save_path}/real")
        writer_fake = SummaryWriter(f"{self.save_path}/fake")

        writer_disc_loss = SummaryWriter(f"{self.save_path}/disc/loss")  # track disc and gen loss
        writer_gen_loss = SummaryWriter(f"{self.save_path}/gen/loss")

        writer_disc_prob = SummaryWriter(f"{self.save_path}/disc/prob")
        # track D(x)  - prob. of correctly classifing real img
        writer_gen_prob = SummaryWriter(f"{self.save_path}/gen/prob")
        # track D(G(z)) - prob of correctly classifing generated img)

        step = 0

        gen.train()
        disc.train()

        for epoch in range(self.num_epochs):
            # don't need targets
            for batch_idx, (real, _) in enumerate(dataloader):
                real = real.to(device)

                # generate noise
                noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
                fake = gen(noise)

                # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
                disc_real = disc(real).reshape(-1)

                D_x = disc_real.mean().item()
                loss_disc_real = loss(disc_real, torch.ones_like(disc_real))
                disc_fake = disc(fake.detach()).reshape(-1)
                loss_disc_fake = loss(disc_fake, torch.zeros_like(disc_fake))
                loss_disc = (loss_disc_real + loss_disc_fake) / 2
                disc.zero_grad()
                loss_disc.backward()
                opt_disc.step()

                # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))

                output = disc(fake).reshape(-1)

                D_G_z = output.mean().item()
                # mean classification for fake
                loss_gen = loss(output, torch.ones_like(output))
                gen.zero_grad()
                loss_gen.backward()
                opt_gen.step()

                # Print losses occasionally and print to tensorboard
                # ideal setup is D(X) and D(G(Z)) to both be at .5,
                # meaning the discriminator has a 50% chance of guessing
                # whether it's a real or fake image

                if self.verbose != 0 and batch_idx % 100 == 0:
                    print(f"Epoch [{epoch}/{self.num_epochs}] Batch {batch_idx}/{len(dataloader)}"
                          f"Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}, D_X: {D_x}, D_G_z: {D_G_z}")

                    with torch.no_grad():
                        fake = gen(fixed_noise)
                        # take out (up to) 32 examples
                        img_grid_real = torchvision.utils.make_grid(
                            real[:32], normalize=True
                        )
                        img_grid_fake = torchvision.utils.make_grid(
                            fake[:32], normalize=True
                        )

                        writer_real.add_image("Real", img_grid_real, global_step=step)
                        writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                        writer_disc_loss.add_scalar("disc/loss", loss_disc, global_step=step)
                        writer_gen_loss.add_scalar("gen/loss", loss_gen, global_step=step)

                        writer_disc_prob.add_scalar("disc/prob", D_x, global_step=step)
                        writer_gen_prob.add_scalar("gen/prob", D_G_z, global_step=step)

                    step += 1
