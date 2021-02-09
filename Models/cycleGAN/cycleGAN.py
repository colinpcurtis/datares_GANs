from torch import nn
from torch.cuda import is_available
from torch.optim import Adam
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import ones_like, no_grad, save, device, mean, abs

# needed to preprocess
from config import PROJECT_ROOT
import os
from PIL import ImageFile

# reuse pix2pix generator and discriminator for architectures
from Models.ConditionalGAN.uNetGenerator import conditionalGenerator
from Models.cycleGAN.cycleDiscriminator import cycleDiscriminator

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


class cycleGAN:
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

    def discriminator_loss(self, loss_fn, real, generated):
        """
            Args:
                loss_fn: loss function
                real: real discriminator output
                generated: discriminator output from generated image
            Returns:
                sum of losses for real and fake results
        """
        real_loss = loss_fn(real, ones_like(real).detach())
        gen_loss = loss_fn(generated, ones_like(generated).detach())
        return real_loss + gen_loss

    def generator_loss(self, loss_fn, generated):
        """
            Args:
                loss_fn: loss function
                generated: discriminator output from generated images
            Returns:
                loss function applied to generated
        """
        return loss_fn(generated, ones_like(generated).detach())

    def cycle_loss(self, real_im, cycle_im):
        """
            Args:
                real_im: real image, eg (A)
                cycle_im: real image cycled through both generators, eg G(F(A))
            Returns:
                L1 norm between real_im and cycle_im, want cycled image to be close to original
        """
        return mean(abs(real_im - cycle_im))

    def identity_loss(self, real_im, same_im):
        """
            Args:
                real_im: real image, eg (A)
                same_im: image passed through generator, eg G(A)
            Returns:
                L1 norm between real_im and same_im , want generated image to be close to original
        """
        return mean(abs(real_im - same_im))

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
            Runs training session of cycle GAN
        """
        dataset = self.dataset("/images")
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

        # generator G learns mapping G: A -> B
        genG = conditionalGenerator(channels_img=CHANNELS_IMG).to(device)
        # discriminator G differentiates between A and F(B)
        discG = cycleDiscriminator(channels_img=CHANNELS_IMG).to(device)

        # generator F learns mapping F: B -> A
        genF = conditionalGenerator(channels_img=CHANNELS_IMG).to(device)
        # discriminator F differentiates between B and G(A)
        discF = cycleDiscriminator(channels_img=CHANNELS_IMG).to(device)

        writer_real = SummaryWriter(f"logs/{self.save_path_logs}/real")
        writer_fake = SummaryWriter(f"logs/{self.save_path_logs}/fake")

        writer_disc_loss = SummaryWriter(f"logs/{self.save_path_logs}/disc/loss")  # track disc and gen loss
        writer_gen_loss = SummaryWriter(f"logs/{self.save_path_logs}/gen/loss")

        writer_disc_prob = SummaryWriter(f"logs/{self.save_path_logs}/disc/prob")
        # track D(x)  - probability of classifying real image as real
        writer_gen_prob = SummaryWriter(f"logs/{self.save_path_logs}/gen/prob")
        # track D(G(x)) - probability of classifying fake image as real

        # TODO: add LR scheduler
        genG_optimizer = Adam(genG.parameters(), lr=LEARNING_RATE, betas=BETAS)
        discG_optimizer = Adam(discG.parameters(), lr=LEARNING_RATE, betas=BETAS)

        genF_optimizer = Adam(genF.parameters(), lr=LEARNING_RATE, betas=BETAS)
        discF_optimizer = Adam(discF.parameters(), lr=LEARNING_RATE, betas=BETAS)

        loss = nn.BCELoss()

        genG.train()
        discG.train()

        genF.train()
        discF.train()

        step = 0
        # train loop
        for epoch in range(self.num_epochs):
            for batch_id, (imageA_real, imageB_real) in enumerate(dataloader):
                imageA_real = imageA_real.to(device)
                imageB_real = imageB_real.to(device)

                # forward generator
                imageB_fake = genG(imageA_real)
                imageA_fake = genF(imageB_real)

                # cycle back to original image
                cycleA = genF(imageB_fake)
                cycleB = genG(imageA_fake)

                # same is used to generate identity loss
                sameA = genG(imageA_real)
                sameB = genF(imageB_real)

                # forward discriminator
                discA_real = discF(imageA_real)
                discB_real = discG(imageB_real)

                discA_fake = discF(imageA_fake)
                discB_fake = discG(imageB_fake)

                # get probability that discriminator predicts real image as real
                # imageA_prob = discG_pred.mean().item()
                # imageB_prob = discF_pred.mean().item()

                # calculate BCE between real images and what it should output (a vector of ones)
                # and the fake images and what it should output (a vector of zeroes)
                loss_discA = self.discriminator_loss(loss, discA_fake, discA_real)
                loss_discB = self.discriminator_loss(loss, discB_fake, discB_real)

                # generator loss
                genG_loss = self.generator_loss(loss, discA_fake)
                genF_loss = self.generator_loss(loss, discB_fake)

                # cycle loss -> feature matching
                cycle_loss = self.cycle_loss(imageA_real, cycleA) + self.cycle_loss(imageB_real, cycleB)

                # total losses
                total_genG_loss = genG_loss + cycle_loss + self.identity_loss(imageA_real, sameA)
                total_genF_loss = genF_loss + cycle_loss + self.identity_loss(imageB_real, sameB)

                # zero gradients before backward
                genG.zero_grad()
                discG.zero_grad()

                genF.zero_grad()
                discF.zero_grad()

                # calculate gradients
                total_genG_loss.backward(retain_graph=True)
                total_genF_loss.backward(retain_graph=True)

                loss_discA.backward(retain_graph=True)
                loss_discB.backward(retain_graph=True)

                # backpropagate gradients
                genG_optimizer.step()
                genF_optimizer.step()

                discG_optimizer.step()
                discF_optimizer.step()

                # if batch_id % 100 == 0:
                #     print(f"epoch: {epoch}/{self.num_epochs} batch: {batch_id}/{len(dataloader)} "
                #           f"loss D: {total_disc_loss:.4f} loss G: {total_gen_loss:.4f} "
                #           f"D(G(x)): {d_g_x:.4f} D(x): {d_x:.4f}")
                #
                #     with no_grad():
                #         # plot generated and real images
                #         img_grid_real = torchvision.utils.make_grid(imageA[:64], normalize=True)
                #         img_grid_fake = torchvision.utils.make_grid(fake[:64], normalize=True)
                #
                #         writer_real.add_image("Ground Truth", img_grid_real, global_step=step)
                #         writer_fake.add_image("Generated", img_grid_fake, global_step=step)
                #
                #         writer_disc_loss.add_scalar("disc/loss", total_disc_loss, global_step=step)
                #         writer_gen_loss.add_scalar("gen/loss", total_gen_loss, global_step=step)
                #
                #         writer_disc_prob.add_scalar("disc/prob", d_x, global_step=step)
                #         writer_gen_prob.add_scalar("gen/prob", d_g_x, global_step=step)
                #     step += 1
