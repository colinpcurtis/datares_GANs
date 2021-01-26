from torch import nn
from torch import device
from torch.cuda import is_available
from torch.optim import Adam
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import ones_like, zeros_like, abs, no_grad

# needed to preprocess
from config import PROJECT_ROOT
import os
from numpy import asarray
import numpy as np
from PIL import Image

# architectures
from uNetGenerator import conditionalGenerator
from uNetDiscriminator import conditionalDiscriminator

device = device("cuda" if is_available() else "cpu")


# model constants
BATCH_SIZE = 256  # make batch size as big as possible on your machine until you get memory errors
IMAGE_SIZE = 511
CHANNELS_IMG = 3

# hyperparameters
LEARNING_RATE = 1e-3
LAMBDA = 100  # L1 penalty
BETAS = (0.9, 0.999)  # moving average for ADAM


def preprocess(directory):
    """
        convert raw images to 8 bit RGB colors and replace raw image
        with converted one
    """
    root = PROJECT_ROOT
    img_root = os.fsencode(root + directory)
    # loop through dataset dirs and subdirs
    for subdir in os.listdir(img_root):
        folder = os.path.join(img_root, subdir)
        try:
            for file in os.listdir(folder):
                filename = os.fsdecode(file)
                file_path = os.path.join(os.fsdecode(folder), filename)
                if file_path.endswith(".png"):
                    image = Image.open(file_path)
                    data = asarray(image)
                    img = Image.fromarray(np.uint8(data)).convert('RGB')
                    img.save(file_path)
                    # img.show()
        except NotADirectoryError:
            # could try deleting directories with error
            print(folder, "ERROR DIRECTORY")


class conditionalGAN:
    """
        implements conditional GAN training
    """
    def __init__(self, num_epochs, save_dir, save_path):
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.save_path = save_path

    def transform(self):
        # TODO: find better normalization method
        data_transforms = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)],
                                                                   [0.5 for _ in range(CHANNELS_IMG)])])
        return data_transforms

    def dataset(self, directory: str):
        root = PROJECT_ROOT
        img_root = os.fsdecode(root + directory)
        images = ImageFolder(root=img_root, transform=self.transform())
        return images

    def train(self):
        dataset = self.dataset("/test_images")
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        gen = conditionalGenerator(channels_img=CHANNELS_IMG).to(device)
        disc = conditionalDiscriminator(channels_img=CHANNELS_IMG).to(device)

        writer_real = SummaryWriter(f"logs/{self.save_path}/real")
        writer_fake = SummaryWriter(f"logs/{self.save_path}/fake")

        writer_disc_loss = SummaryWriter(f"logs/{self.save_path}/disc/loss")  # track disc and gen loss
        writer_gen_loss = SummaryWriter(f"logs/{self.save_path}/gen/loss")

        writer_disc_prob = SummaryWriter(f"logs/{self.save_path}/disc/prob")
        # track D(x)  - probability of classifying real image as real
        writer_gen_prob = SummaryWriter(f"logs/{self.save_path}/gen/prob")
        # track D(G(x)) - probability of classifying fake image as real

        # we'll want to add an LR scheduler eventually
        gen_optimizer = Adam(gen.parameters(), lr=LEARNING_RATE, betas=BETAS)
        disc_optimizer = Adam(disc.parameters(), lr=LEARNING_RATE, betas=BETAS)

        loss = nn.BCELoss()

        # add L1 regularization on generator to promote feature matching and image clearness
        l1_loss = nn.L1Loss()

        gen.train()
        disc.train()

        step = 0
        # train loop
        for epoch in range(self.num_epochs):
            for batch_id, (real, _) in enumerate(dataloader):
                fake = gen(real)

                # train discriminator

                disc_real = disc(real)  # feed real image to discriminator
                disc_fake = disc(fake)  # feed fake image to discriminator

                d_x = disc_real.mean().item()  # get probability that discriminator predicts real image as real
                d_g_x = disc_fake.mean().item()  # probability of fake image classified as real

                # calculate BCE between real images and what it should output (a vector of ones)
                # and the fake images and what it should output (a vector of zeroes)
                loss_disc_real = loss(ones_like(disc_real), disc_real)
                loss_disc_fake = loss(zeros_like(disc_fake.detatch()), disc_fake)
                # TODO: check if fake images should be detached

                total_disc_loss = loss_disc_real + loss_disc_fake

                # train generator

                # calculate BCE between fake images and what generator should output (a vector of ones)
                gen_loss = loss(ones_like(fake), fake)
                gen_l1_loss = l1_loss(abs(real - fake))

                total_gen_loss = gen_loss + (LAMBDA * gen_l1_loss)

                # zero gradients before backward
                gen.zero_grad()
                disc.zero_grad()

                # calculate gradients
                total_gen_loss.backward()
                total_disc_loss.backward()

                # backpropagate gradients
                gen_optimizer.step()
                disc_optimizer.step()

                if batch_id % 100 == 0:
                    print(f"epoch: {epoch}/{self.num_epochs} batch: {batch_id}/{len(dataloader)} "
                          f"loss D: {total_disc_loss:.4f} loss G: {total_gen_loss:.4f} "
                          f"D(G(x)): {d_g_x:.4f} D(x): {d_x:.4f}")

                    with no_grad():
                        # plot generated and real images
                        img_grid_real = torchvision.utils.make_grid(real[:64], normalize=True)
                        img_grid_fake = torchvision.utils.make_grid(fake[:64], normalize=True)

                        writer_real.add_image("Ground Truth", img_grid_real, global_step=step)
                        writer_fake.add_image("Generated", img_grid_fake, global_step=step)

                        writer_disc_loss.add_scalar("disc/loss", total_disc_loss, global_step=step)
                        writer_gen_loss.add_scalar("gen/loss", total_gen_loss, global_step=step)

                        writer_disc_prob.add_scalar("disc/prob", d_x, global_step=step)
                        writer_gen_prob.add_scalar("gen/prob", d_g_x, global_step=step)

                    step += 1
