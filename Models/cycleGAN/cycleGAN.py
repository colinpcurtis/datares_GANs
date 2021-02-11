from torch import nn
from torch.cuda import is_available
from torch.optim import Adam
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torch import ones_like, zeros_like, no_grad, save, device, mean, abs, randn

# needed to preprocess
from config import PROJECT_ROOT
import os
from PIL import ImageFile

# reuse pix2pix generator and discriminator for architectures
from Models.ConditionalGAN.uNetGenerator import conditionalGenerator
from Models.cycleGAN.cycleDiscriminator import cycleDiscriminator

device = device("cuda" if is_available() else "cpu")


# model constants
BATCH_SIZE = 3  # make batch size as big as possible on your machine until you get memory errors
IMAGE_SIZE = 511
CHANNELS_IMG = 3
ImageFile.LOAD_TRUNCATED_IMAGES = True

# hyperparameters
LEARNING_RATE = 1e-4
LAMBDA = 10  # L1 penalty
BETAS = (0.9, 0.999)  # moving average for ADAM


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class cycleGAN:
    """
        implements conditional GAN
    """
    def __init__(self, num_epochs, save_path_logs, save_path_model, dataset_dir):
        """
            Args:
                num_epochs: number of epochs to train model
                save_path_logs: path to save tensorboard logs from /logs directory in project
                save_path_model: path to save model state dict at end of training
        """
        self.num_epochs = num_epochs
        self.save_path_logs = save_path_logs
        self.save_path_model = save_path_model
        self.dataset_dir = dataset_dir

    def transform(self, img_mean, img_std):
        """
            Args:
                self
            Returns:
                Data transform composition
        """
        # TODO: calculate mean and std of data
        data_transforms = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                              transforms.ToTensor(),
                                              transforms.Normalize([img_mean for _ in range(CHANNELS_IMG)],
                                                                   [img_std for _ in range(CHANNELS_IMG)]),
                                              AddGaussianNoise(img_mean, .05)])
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
        # imagesA = ImageFolder(root=f"{img_root}", transform=self.transform())
        imagesA = ImageFolder(root=f"{img_root}/imsA/", transform=self.transform(.511, .227))
        imagesB = ImageFolder(root=f"{img_root}/imsB/", transform=self.transform(.413, .262))
        return imagesA, imagesB

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
        gen_loss = loss_fn(generated, zeros_like(generated).detach())
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
                L1 norm between real_im and same_im, promotes feature matching for generator
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
        # A is paintings, B is photos
        imagesA, imagesB = self.dataset(self.dataset_dir)

        dataloader1 = DataLoader(imagesA, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
        dataloader2 = DataLoader(imagesB, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)

        # generator G learns mapping G: A -> B
        genG = conditionalGenerator(channels_img=CHANNELS_IMG).to(device)
        # discriminator G differentiates between A and F(B)
        discG = cycleDiscriminator(channels_img=CHANNELS_IMG).to(device)

        # generator F learns mapping F: B -> A
        genF = conditionalGenerator(channels_img=CHANNELS_IMG).to(device)
        # discriminator F differentiates between B and G(A)
        discF = cycleDiscriminator(channels_img=CHANNELS_IMG).to(device)

        writer_realA = SummaryWriter(f"{PROJECT_ROOT}/logs/{self.save_path_logs}/A/real")
        writer_fakeA = SummaryWriter(f"{PROJECT_ROOT}/logs/{self.save_path_logs}/A/fake")

        writer_realB = SummaryWriter(f"{PROJECT_ROOT}/logs/{self.save_path_logs}/B/real")
        writer_fakeB = SummaryWriter(f"{PROJECT_ROOT}/logs/{self.save_path_logs}/B/fake")

        writer_disc_lossG = SummaryWriter(f"{PROJECT_ROOT}/logs/{self.save_path_logs}/disc/lossG")  # track disc and gen loss
        writer_gen_lossG = SummaryWriter(f"{PROJECT_ROOT}/logs/{self.save_path_logs}/gen/lossG")

        writer_disc_lossF = SummaryWriter(f"{PROJECT_ROOT}/logs/{self.save_path_logs}/disc/lossF")  # track disc and gen loss
        writer_gen_lossF = SummaryWriter(f"{PROJECT_ROOT}/logs/{self.save_path_logs}/gen/lossF")

        # TODO: add LR scheduler
        genG_optimizer = Adam(genG.parameters(), lr=LEARNING_RATE, betas=BETAS)
        discG_optimizer = Adam(discG.parameters(), lr=LEARNING_RATE, betas=BETAS)

        genF_optimizer = Adam(genF.parameters(), lr=LEARNING_RATE, betas=BETAS)
        discF_optimizer = Adam(discF.parameters(), lr=LEARNING_RATE, betas=BETAS)

        lossDiscG = nn.BCEWithLogitsLoss()
        lossDiscF = nn.BCEWithLogitsLoss()

        lossGenG = nn.BCEWithLogitsLoss()
        lossGenF = nn.BCEWithLogitsLoss()

        genG.train()
        discG.train()

        genF.train()
        discF.train()

        step = 0
        # train loop
        for epoch in range(self.num_epochs):
            for batch_id, (imageA_real, imageB_real) in enumerate(zip(dataloader1, dataloader2)):
                if min(len(dataloader2), len(dataloader1)) <= batch_id:
                    # the dataloaders are not the same size
                    break
                # zip concats to list of lists, so need to unpack to get raw tensors
                imageA_real = imageA_real[0].to(device)
                imageB_real = imageB_real[0].to(device)

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
                loss_discG = self.discriminator_loss(lossDiscG, discA_fake, discA_real)
                loss_discF = self.discriminator_loss(lossDiscF, discB_fake, discB_real)

                # generator loss
                genG_loss = self.generator_loss(lossGenG, discA_fake)
                genF_loss = self.generator_loss(lossGenF, discB_fake)

                # cycle loss -> feature matching
                cycle_loss = self.cycle_loss(imageA_real, cycleA) + self.cycle_loss(imageB_real, cycleB)

                # total losses
                total_genG_loss = genG_loss + (LAMBDA * cycle_loss) + (LAMBDA * self.identity_loss(imageA_real, sameA))
                total_genF_loss = genF_loss + (LAMBDA * cycle_loss) + (LAMBDA * self.identity_loss(imageB_real, sameB))

                # zero gradients before backward
                genG.zero_grad()
                discG.zero_grad()

                genF.zero_grad()
                discF.zero_grad()

                # calculate gradients
                total_genG_loss.backward(retain_graph=True)
                total_genF_loss.backward(retain_graph=True)

                loss_discG.backward(retain_graph=True)
                loss_discF.backward(retain_graph=True)

                # backpropagate gradients
                genG_optimizer.step()
                genF_optimizer.step()
                # only update disc if gen has gotten stronger
                if loss_discG > .5:
                    discG_optimizer.step()
                if loss_discF > .5:
                    discF_optimizer.step()

                if batch_id % 10 == 0:
                    print(f"epoch: {epoch}/{self.num_epochs} "
                          f"batch: {batch_id}/{min(len(dataloader1), len(dataloader2))} "
                          f"disc loss G: {loss_discG:.4f} disc loss F: {loss_discF:.4f} "
                          f"gen loss G: {total_genG_loss:.4f} gen loss F: {total_genF_loss:.4f}")

                    with no_grad():
                        # plot generated and real images
                        img_grid_realA = torchvision.utils.make_grid(imageA_real[:16], normalize=True)
                        img_grid_fakeA = torchvision.utils.make_grid(imageA_fake[:16], normalize=True)

                        img_grid_realB = torchvision.utils.make_grid(imageB_real[:16], normalize=True)
                        img_grid_fakeB = torchvision.utils.make_grid(imageB_fake[:16], normalize=True)

                        writer_realA.add_image("Ground Truth A", img_grid_realA, global_step=step)
                        writer_fakeA.add_image("Generated A", img_grid_fakeA, global_step=step)

                        writer_realB.add_image("Ground Truth B", img_grid_realB, global_step=step)
                        writer_fakeB.add_image("Generated B", img_grid_fakeB, global_step=step)

                        writer_disc_lossG.add_scalar("disc/lossDiscG", loss_discG, global_step=step)
                        writer_gen_lossG.add_scalar("gen/lossGenG", total_genG_loss, global_step=step)

                        writer_disc_lossF.add_scalar("disc/lossDiscF", loss_discF, global_step=step)
                        writer_gen_lossF.add_scalar("gen/lossGenF", total_genF_loss, global_step=step)

                    step += 1


if __name__ == "__main__":
    # simple testing script
    gan = cycleGAN(1, None, None, "/monet2photo")
    gan.train()

