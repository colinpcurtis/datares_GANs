from torch import nn
from torch.cuda import is_available
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch

# needed to preprocess
from config import PROJECT_ROOT
import os
from PIL import ImageFile

# reuse pix2pix generator and discriminator for architectures
from Models.cycleGAN.cycleGenerator import cycleGenerator
from Models.cycleGAN.cycleDiscriminator import cycleDiscriminator

torch.manual_seed(42)

# model constants
BATCH_SIZE = 3  # make batch size as big as possible on your machine until you get memory errors
IMAGE_SIZE = 511
CHANNELS_IMG = 3
ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device("cuda" if is_available() else "cpu")

# hyperparameters
LEARNING_RATE = 1e-2
LAMBDA = 1  # L1 penalty
BETAS = (0.9, 0.999)  # moving average for ADAM
GAUSSIAN_NOISE_STD = .05
SCHEDULER_STEP_SIZE = 100
GAMMA = 0.96


class AddGaussianNoise(object):
    """
        Add gaussian noise with specified mean and standard deviation to an input tensor
    """
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class cycleGAN:
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

    def transform(self):
        """
            Args:
                self
            Returns:
                Data transform composition
        """
        data_transforms = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                              transforms.ToTensor(),
                                              transforms.Normalize(.5, .5),
                                              AddGaussianNoise(0, GAUSSIAN_NOISE_STD)])
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
        imagesA = ImageFolder(root=f"{img_root}/imsA/", transform=self.transform())
        imagesB = ImageFolder(root=f"{img_root}/imsB/", transform=self.transform())
        return imagesA, imagesB

    def gan_loss(self, target, generated):
        """
            Args:
                generated: generated images
                target: image to match
            Returns:
                loss function applied to generated
        """
        return nn.MSELoss()(target, generated)

    def cycle_loss(self, real_im, cycle_im):
        """
            Args:
                real_im: real image, eg (A)
                cycle_im: real image cycled through both generators, eg G(F(A))
            Returns:
                L1 norm between real_im and cycle_im, want cycled image to be close to original
        """
        return nn.L1Loss()(real_im - cycle_im)

    def identity_loss(self, real_im, same_im):
        """
            Args:
                real_im: real image, eg (A)
                same_im: image passed through generator, eg G(A)
            Returns:
                L1 norm between real_im and same_im, promotes feature matching for generator
        """
        return nn.L1Loss()(real_im - same_im)

    def save_model(self, model, save_path):
        """
            Args:
                save_path: path from project root to save model state dict (use .pt extension)
            Returns:
                pickle file at at path with model state dict
        """
        torch.save(model.state_dict(), save_path)

    def train(self):
        """
            Runs training session of cycle GAN
        """
        # A is paintings, B is photos
        imagesA, imagesB = self.dataset("/datasets" + self.dataset_dir)

        dataloader1 = DataLoader(imagesA, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
        dataloader2 = DataLoader(imagesB, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)

        # generator G learns mapping G: A -> B
        genA2B = cycleGenerator(image_size=CHANNELS_IMG).to(device)
        # discriminator G differentiates between A and F(B)
        discB = cycleDiscriminator(channels_img=CHANNELS_IMG).to(device)

        # generator F learns mapping F: B -> A
        genB2A = cycleGenerator(image_size=CHANNELS_IMG).to(device)
        # discriminator F differentiates between B and G(A)
        discA = cycleDiscriminator(channels_img=CHANNELS_IMG).to(device)

        writer_realA = SummaryWriter(f"{PROJECT_ROOT}/logs/{self.save_path_logs}/A/real")
        writer_fakeA = SummaryWriter(f"{PROJECT_ROOT}/logs/{self.save_path_logs}/A/fake")

        writer_realB = SummaryWriter(f"{PROJECT_ROOT}/logs/{self.save_path_logs}/B/real")
        writer_fakeB = SummaryWriter(f"{PROJECT_ROOT}/logs/{self.save_path_logs}/B/fake")

        writer_disc_lossG = SummaryWriter(f"{PROJECT_ROOT}/logs/{self.save_path_logs}/disc/lossG")  # track disc and gen loss
        writer_gen_lossG = SummaryWriter(f"{PROJECT_ROOT}/logs/{self.save_path_logs}/gen/lossG")

        writer_disc_lossF = SummaryWriter(f"{PROJECT_ROOT}/logs/{self.save_path_logs}/disc/lossF")  # track disc and gen loss
        writer_gen_lossF = SummaryWriter(f"{PROJECT_ROOT}/logs/{self.save_path_logs}/gen/lossF")

        # optimizers
        genA2B_optimizer = Adam(genA2B.parameters(), lr=LEARNING_RATE, betas=BETAS)
        discB_optimizer = Adam(discB.parameters(), lr=LEARNING_RATE, betas=BETAS)
        genB2A_optimizer = Adam(genB2A.parameters(), lr=LEARNING_RATE, betas=BETAS)
        discA_optimizer = Adam(discA.parameters(), lr=LEARNING_RATE, betas=BETAS)

        # LR schedulers for corresponding optimizers
        genA2B_scheduler = StepLR(genA2B_optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=GAMMA)
        genB2A_scheduler = StepLR(genB2A_optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=GAMMA)
        discB_scheduler = StepLR(discB_optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=GAMMA)
        discA_scheduler = StepLR(discA_optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=GAMMA)

        genA2B.train()
        discB.train()
        genB2A.train()
        discA.train()

        step = 0
        # train loop
        for epoch in range(self.num_epochs):
            for batch_id, (imageA_real, imageB_real) in enumerate(zip(dataloader1, dataloader2)):
                if min(len(dataloader2), len(dataloader1)) <= batch_id:
                    # the dataloaders are not the same size
                    break
                imageA_real = imageA_real[0].to(device)
                imageB_real = imageB_real[0].to(device)

                # cycle images
                sameA = genA2B(imageA_real)
                sameB = genB2A(imageB_real)

                # forward generator
                imageB_fake = genA2B(imageA_real)
                cycleA = genB2A(imageB_fake)
                imageA_fake = genB2A(imageB_real)
                cycleB = genA2B(imageA_fake)

                # get discriminator predictions
                discA_fake = discA(imageA_fake)
                discB_fake = discB(imageB_fake)

                # generator loss
                target = torch.ones_like(discA_fake).detach()
                lossA2B = self.gan_loss(discA_fake, target)
                lossB2A = self.gan_loss(discB_fake, target)

                # cycle loss -> feature matching
                cycle_loss = self.cycle_loss(imageA_real, cycleA) + self.cycle_loss(imageB_real, cycleB)

                # total losses
                total_genA2B_loss = lossA2B + (LAMBDA * cycle_loss) + (LAMBDA * self.identity_loss(imageA_real, sameA))
                total_genB2A_loss = lossB2A + (LAMBDA * cycle_loss) + (LAMBDA * self.identity_loss(imageB_real, sameB))

                # calculate gen gradients
                genA2B_optimizer.zero_grad()
                genB2A_optimizer.zero_grad()
                total_genA2B_loss.backward()
                total_genB2A_loss.backward()

                genA2B_optimizer.step()
                genB2A_optimizer.step()

                genA2B_scheduler.step()
                genB2A_scheduler.step()

                discA_optimizer.zero_grad()
                discB_optimizer.zero_grad()

                # update discA
                pred_real = discA(imageA_real)
                lossDA_real = self.gan_loss(pred_real, torch.ones_like(pred_real).detach())
                pred_fake = discA(imageA_fake)
                lossDA_fake = self.gan_loss(pred_fake, torch.zeros_like(pred_fake).detach())

                totalDA_loss = (lossDA_real + lossDA_fake) * LAMBDA
                totalDA_loss.backward()
                discA_optimizer.step()

                # update discB
                pred_real = discB(imageB_real)
                lossDB_real = self.gan_loss(pred_real, torch.ones_like(pred_real).detach())
                pred_fake = discB(imageB_fake)
                lossDB_fake = self.gan_loss(pred_fake, torch.zeros_like(pred_fake).detach())
                totalDB_loss = (lossDB_real + lossDB_fake) * LAMBDA
                totalDB_loss.backward()
                discB_optimizer.step()

                discB_scheduler.step()
                discA_scheduler.step()
                if batch_id % 10 == 0:
                    print(f"epoch: {epoch}/{self.num_epochs} "
                          f"batch: {batch_id}/{min(len(dataloader1), len(dataloader2))} "
                          f"disc loss A: {totalDA_loss:.4f} "
                          f"disc loss B: {totalDA_loss:.4f} "
                          f"gen loss A2B: {total_genA2B_loss:.4f} "
                          f"gen loss B2A: {total_genB2A_loss:.4f}")

                    with torch.no_grad():
                        # plot generated and real images
                        img_grid_realA = torchvision.utils.make_grid(imageA_real[:16], normalize=True)
                        img_grid_fakeA = torchvision.utils.make_grid(imageA_fake[:16], normalize=True)

                        img_grid_realB = torchvision.utils.make_grid(imageB_real[:16], normalize=True)
                        img_grid_fakeB = torchvision.utils.make_grid(imageB_fake[:16], normalize=True)

                        writer_realA.add_image("Ground Truth A", img_grid_realA, global_step=step)
                        writer_fakeA.add_image("Generated A", img_grid_fakeA, global_step=step)

                        writer_realB.add_image("Ground Truth B", img_grid_realB, global_step=step)
                        writer_fakeB.add_image("Generated B", img_grid_fakeB, global_step=step)

                        writer_disc_lossG.add_scalar("disc/lossDiscB", totalDB_loss, global_step=step)
                        writer_gen_lossG.add_scalar("gen/lossGenA2B", total_genA2B_loss, global_step=step)

                        writer_disc_lossF.add_scalar("disc/lossDiscA", totalDA_loss, global_step=step)
                        writer_gen_lossF.add_scalar("gen/lossGenB2A", total_genB2A_loss, global_step=step)

                    step += 1

        # save generator from photos to paintings at end of training
        self.save_model(genB2A, self.save_path_model)
