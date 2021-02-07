from torch import nn
from torch.cuda import is_available
from torch.optim import Adam
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import ones_like, zeros_like, no_grad, save, device

# needed to preprocess
from config import PROJECT_ROOT
import os
from PIL import ImageFile

# architectures
from Models.ConditionalGAN.uNetGenerator import conditionalGenerator
from Models.ConditionalGAN.uNetDiscriminator import conditionalDiscriminator

device = device("cuda" if is_available() else "cpu")


# model constants
BATCH_SIZE = 16  # make batch size as big as possible on your machine until you get memory errors
IMAGE_SIZE = 511
CHANNELS_IMG = 3
ImageFile.LOAD_TRUNCATED_IMAGES = True

# hyperparameters
LEARNING_RATE = 1e-3
LAMBDA = 100  # L1 penalty
BETAS = (0.9, 0.999)  # moving average for ADAM


class conditionalGAN:
    """
        implements conditional GAN
    """
    def __init__(self, num_epochs, save_path_logs, save_path_model):
        """
            Args:
                num_epochs: number of epochs to train model
                save_path_logs: path to save tensorboard logs from /logs directory in project
                save_path_model: path to save model state dict at end of training
        """
        self.num_epochs = num_epochs
        self.save_path_logs = save_path_logs
        self.save_path_model = save_path_model

    def transform(self):
        """
            Args:
                self
            Returns:
                Data transform composition
        """
        # TODO: calculate mean and std of data
        data_transforms = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)],
                                                                   [0.5 for _ in range(CHANNELS_IMG)])])
        return data_transforms

    def dataset(self, directory: str):
        """
            Args:
                directory: path from project root to image folders:
            Returns:
                transformed image dataset
        """
        root = PROJECT_ROOT
        img_root = os.fsdecode(root + directory)
        images = ImageFolder(root=img_root, transform=self.transform())
        return images

    def save_model(self, state_dict, save_path):
        """
            Args:
                save_path: path from project root to save model state dict (use .pt extension)
                state_dict: dictionary storing model params
            Returns:
                pickle file at at path with model state dict
        """
        save(state_dict, save_path)

    def train(self):
        """
            Runs training session of conditional GAN
        """
        dataset = self.dataset("/images")
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

        gen = conditionalGenerator(channels_img=CHANNELS_IMG).to(device)
        disc = conditionalDiscriminator(channels_img=CHANNELS_IMG).to(device)

        writer_real = SummaryWriter(f"logs/{self.save_path_logs}/real")
        writer_fake = SummaryWriter(f"logs/{self.save_path_logs}/fake")

        writer_disc_loss = SummaryWriter(f"logs/{self.save_path_logs}/disc/loss")  # track disc and gen loss
        writer_gen_loss = SummaryWriter(f"logs/{self.save_path_logs}/gen/loss")

        writer_disc_prob = SummaryWriter(f"logs/{self.save_path_logs}/disc/prob")
        # track D(x)  - probability of classifying real image as real
        writer_gen_prob = SummaryWriter(f"logs/{self.save_path_logs}/gen/prob")
        # track D(G(x)) - probability of classifying fake image as real

        # TODO: add LR scheduler
        gen_optimizer = Adam(gen.parameters(), lr=LEARNING_RATE, betas=BETAS)
        disc_optimizer = Adam(disc.parameters(), lr=LEARNING_RATE, betas=BETAS)

        loss = nn.BCELoss()

        # add MSE on generator to promote feature matching and image clearness
        l1_loss = nn.L1Loss()
        # just want this to add a penalty for not adding images - no backward
        l1_loss.requires_grad = False

        gen.train()
        disc.train()

        step = 0
        # train loop
        for epoch in range(self.num_epochs):
            for batch_id, (real, _) in enumerate(dataloader):
                real = real.to(device)
                fake = gen(real)

                # train discriminator
                disc_real = disc(real, real)
                disc_fake = disc(real, fake)

                d_x = disc_real.mean().item()  # get probability that discriminator predicts real image as real
                d_g_x = disc_fake.mean().item()  # probability of fake image classified as real

                # calculate BCE between real images and what it should output (a vector of ones)
                # and the fake images and what it should output (a vector of zeroes)
                loss_disc_real = loss(disc_real, ones_like(disc_real).detach())
                loss_disc_fake = loss(disc_fake, zeros_like(disc_fake).detach())

                total_disc_loss = loss_disc_real + loss_disc_fake

                # train generator

                # calculate BCE between fake images and what generator should output (a vector of ones)
                gen_loss = loss(disc_fake, ones_like(disc_fake).detach())

                # calculate L1 norm between real and fake image -> promote feature matching
                # this is not a trainable loss just a generator penalty
                gen_l1_loss = l1_loss(fake, real)
                # check: should real be detached

                total_gen_loss = gen_loss + (LAMBDA * gen_l1_loss)

                # zero gradients before backward
                gen.zero_grad()
                disc.zero_grad()

                # calculate gradients
                # the computation graphs share parts in common so save the graphs
                total_gen_loss.backward(retain_graph=True)
                total_disc_loss.backward(retain_graph=True)

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
