from torch import nn
from torch.cuda import is_available
from torch.optim import Adam
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import ones_like, zeros_like, device, randn, manual_seed

# needed to preprocess
from config import PROJECT_ROOT
import os
from PIL import ImageFile

# reuse pix2pix generator and discriminator for architectures
from Models.cycleGAN.CycleDiscriminator import CycleDiscriminator

manual_seed(42)

# model constants
BATCH_SIZE = 50
IMAGE_SIZE = 511
CHANNELS_IMG = 3
ImageFile.LOAD_TRUNCATED_IMAGES = True
device = device("cuda" if is_available() else "cpu")

# hyperparameters
LEARNING_RATE = 1e-3
BETAS = (5, 0.999)  # moving average for ADAM
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
        return tensor + randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class testDisc:
    """
        Simple binary classification between photos and paintings to ensure
        discriminator can learn a decision boundary
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
        data_transforms = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                              transforms.ToTensor(),
                                              transforms.Normalize([img_mean for _ in range(CHANNELS_IMG)],
                                                                   [img_std for _ in range(CHANNELS_IMG)]),
                                              AddGaussianNoise(img_mean, GAUSSIAN_NOISE_STD)])
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

    def train(self):
        """
            Runs training session of cycle GAN
        """
        # A is paintings, B is photos
        imagesA, imagesB = self.dataset("/datasets" + self.dataset_dir)

        dataloader1 = DataLoader(imagesA, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
        dataloader2 = DataLoader(imagesB, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)

        disc = CycleDiscriminator(channels_img=CHANNELS_IMG).to(device)

        optimizer = Adam(disc.parameters(), lr=LEARNING_RATE, betas=BETAS)
        loss_fn = nn.BCEWithLogitsLoss()

        disc.train()

        for epoch in range(self.num_epochs):
            for batch_id, (imageA_real, imageB_real) in enumerate(zip(dataloader1, dataloader2)):
                if min(len(dataloader2), len(dataloader1)) <= batch_id:
                    # the dataloaders are not the same size
                    break

                imageA_real = imageA_real[0].to(device)
                imageB_real = imageB_real[0].to(device)

                outputA = disc(imageA_real)
                outputB = disc(imageB_real)

                disc_loss = self.discriminator_loss(loss_fn, outputA, outputB)

                disc_loss.zero_grad()
                disc_loss.backward()
                optimizer.step()

                if batch_id % 10 == 0:
                    print(f"epoch: {epoch}/{self.num_epochs} "
                          f"batch: {batch_id}/{min(len(dataloader1), len(dataloader2))} "
                          f"disc loss: {disc_loss:.4f}")
