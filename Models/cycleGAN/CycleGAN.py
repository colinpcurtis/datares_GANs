from torch import nn
import torch.nn.functional as F
from torch.cuda import is_available
from torch.optim import Adam
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

from Models.cycleGAN.CycleGenerator import CycleGenerator
from Models.cycleGAN.CycleDiscriminator import CycleDiscriminator

torch.manual_seed(42)  # ensures reproducibility for random initializations

# model constants
BATCH_SIZE = 2  # make batch size as big as possible on your machine until you get memory errors
IMAGE_SIZE = 512
CHANNELS_IMG = 3
ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device("cuda" if is_available() else "cpu")

# hyperparameters
LEARNING_RATE = 2e-4
LAMBDA = 5  # L1 penalty
BETAS = (0.5, 0.999)  # moving average for ADAM
GAUSSIAN_NOISE_STD = .05


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


class CycleGAN:
    def __init__(self, num_epochs, save_path_logs, save_path_model, dataset_dir, reload_trained=True):
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
        self.reload_trained = reload_trained

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
        return F.mse_loss(target, generated)

    def cycle_loss(self, real_im, cycle_im):
        """
            Args:
                real_im: real image, eg (A)
                cycle_im: real image cycled through both generators, eg G(F(A))
            Returns:
                L1 norm between real_im and cycle_im, want cycled image to be close to original
        """
        return F.l1_loss(real_im, cycle_im)

    def identity_loss(self, real_im, same_im):
        """
            Args:
                real_im: real image, eg (A)
                same_im: image passed through generator, eg G(A)
            Returns:
                L1 norm between real_im and same_im, promotes feature matching for generator
        """
        return F.l1_loss(real_im, same_im)

    def train(self):
        """
            Runs training session of cycle GAN
        """
        # A is paintings, B is photos
        imagesA, imagesB = self.dataset("/datasets" + self.dataset_dir)

        dataloader1 = DataLoader(imagesA, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
        dataloader2 = DataLoader(imagesB, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

        if self.reload_trained:
            print("loading pretrained models...")
            # reload pre-trained models for transfer learning
            genA2B = torch.load(f"{PROJECT_ROOT}/{self.save_path_model}/genA2B.pt", map_location=device)
            genB2A = torch.load(f"{PROJECT_ROOT}/{self.save_path_model}/genB2A.pt", map_location=device)
            discA = torch.load(f"{PROJECT_ROOT}/{self.save_path_model}/discA.pt", map_location=device)
            discB = torch.load(f"{PROJECT_ROOT}/{self.save_path_model}/discB.pt", map_location=device)
            disc_optimizer = torch.load(f"{PROJECT_ROOT}/{self.save_path_model}/disc_optimizer.pt", map_location=device)
            gen_optimizer = torch.load(f"{PROJECT_ROOT}/{self.save_path_model}/gen_optimizer.pt", map_location=device)

            # quantize to int8 model for faster training
            genA2B = torch.quantization.quantize_dynamic(genA2B, 
                                                        {torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.InstanceNorm2d}, 
                                                        torch.qint8)
            genB2A = torch.quantization.quantize_dynamic(genB2A, 
                                                        {torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.InstanceNorm2d}, 
                                                        torch.qint8)
            discA = torch.quantization.quantize_dynamic(discA, 
                                                        {torch.nn.Conv2d, torch.nn.InstanceNorm2d}, 
                                                        torch.qint8)
            discB = torch.quantization.quantize_dynamic(discB, 
                                                        {torch.nn.Conv2d, torch.nn.InstanceNorm2d}, 
                                                        torch.qint8)
        else:
            # don't use int8 on initial training because it collapses
            # the model.  Instead train for a few batches on float32
            # then reload those saved models for int8 once there is 
            # a stable baseline established from the generated images

            # generator learns mapping A -> B
            genA2B = CycleGenerator(image_size=CHANNELS_IMG).to(device)
            # discriminator differentiates between A and F(B)
            discB = CycleDiscriminator(channels_img=CHANNELS_IMG).to(device)

            # generator learns mapping  B -> A
            genB2A = CycleGenerator(image_size=CHANNELS_IMG).to(device)
            # discriminator differentiates between B and G(A)
            discA = CycleDiscriminator(channels_img=CHANNELS_IMG).to(device)

            gen_optimizer = Adam(list(genA2B.parameters()) + list(genB2A.parameters()), lr=LEARNING_RATE, betas=BETAS)
            disc_optimizer = Adam(list(discB.parameters()) + list(discA.parameters()), lr=LEARNING_RATE, betas=BETAS)

        writer_realA = SummaryWriter(f"{PROJECT_ROOT}/logs/{self.save_path_logs}/A/real")
        writer_fakeA = SummaryWriter(f"{PROJECT_ROOT}/logs/{self.save_path_logs}/A/fake")

        writer_realB = SummaryWriter(f"{PROJECT_ROOT}/logs/{self.save_path_logs}/B/real")
        writer_fakeB = SummaryWriter(f"{PROJECT_ROOT}/logs/{self.save_path_logs}/B/fake")

        writer_disc_lossG = SummaryWriter(f"{PROJECT_ROOT}/logs/{self.save_path_logs}/disc/lossG")
        writer_gen_lossG = SummaryWriter(f"{PROJECT_ROOT}/logs/{self.save_path_logs}/gen/lossG")

        writer_disc_lossF = SummaryWriter(f"{PROJECT_ROOT}/logs/{self.save_path_logs}/disc/lossF")
        writer_gen_lossF = SummaryWriter(f"{PROJECT_ROOT}/logs/{self.save_path_logs}/gen/lossF")

        
        genA2B.train()
        discB.train()
        genB2A.train()
        discA.train()

        step = 0
        # train loop
        for epoch in range(self.num_epochs):
            for batch_id, (A_real, B_real) in enumerate(zip(dataloader1, dataloader2)):
                if min(len(dataloader2), len(dataloader1)) <= batch_id:
                    # the dataloaders are not the same size
                    break
                A_real = A_real[0].to(device)
                B_real = B_real[0].to(device)

                # train discriminator
                B_fake = genA2B(A_real)
                discA_real = discA(A_real)
                discB_fake = discB(B_fake.detach())
                discA_real_loss = self.gan_loss(discA_real, torch.ones_like(discA_real))
                discA_fake_loss = self.gan_loss(discB_fake, torch.zeros_like(discB_fake))
                discA_loss = discA_real_loss + discA_fake_loss

                A_fake = genB2A(B_real)
                discB_real = discB(A_real)
                discA_fake = discA(A_fake.detach())
                discB_real_loss = self.gan_loss(discB_real, torch.ones_like(discB_real))
                discB_fake_loss = self.gan_loss(discA_fake, torch.zeros_like(discA_fake))
                discB_loss = discB_real_loss + discB_fake_loss

                disc_loss = (discA_loss + discB_loss) / 2
                disc_optimizer.zero_grad()
                disc_loss.backward()
                disc_optimizer.step()

                # train generator

                # get adversarial loss
                discA_fake = discA(A_fake)
                discB_fake = discB(B_fake)
                genA2B_loss = self.gan_loss(discA_fake, torch.ones_like(discA_fake))
                genB2A_loss = self.gan_loss(discB_fake, torch.ones_like(discB_fake))
                adv_loss = genA2B_loss + genB2A_loss

                # cycle loss
                cycleA = genB2A(B_fake)
                cycleB = genA2B(A_fake)
                cycle_loss = self.cycle_loss(cycleA, A_real) + self.cycle_loss(cycleB, B_real)

                # identity loss
                A_id = genA2B(A_real)
                B_id = genB2A(B_real)
                id_loss = self.identity_loss(A_id, A_real) + self.identity_loss(B_id, B_real)

                gen_loss = adv_loss + LAMBDA * (cycle_loss + id_loss)
                gen_optimizer.zero_grad()
                gen_loss.backward()
                gen_optimizer.step()

                best_gen_loss = 10000
                if gen_loss < best_gen_loss:
                    # when our generator gets better then its previous best then we'll checkpoint model
                    torch.save(genA2B, f"{PROJECT_ROOT}/{self.save_path_model}/genA2B.pt")
                    torch.save(genB2A, f"{PROJECT_ROOT}/{self.save_path_model}/genB2A.pt")
                    torch.save(discA, f"{PROJECT_ROOT}/{self.save_path_model}/discA.pt")
                    torch.save(discB, f"{PROJECT_ROOT}/{self.save_path_model}/discB.pt")
                    torch.save(disc_optimizer, f"{PROJECT_ROOT}/{self.save_path_model}/disc_optimizer.pt")
                    torch.save(gen_optimizer, f"{PROJECT_ROOT}/{self.save_path_model}/gen_optimizer.pt")

                    # reset best loss for next checkpoint
                    best_gen_loss = gen_loss

                if batch_id % 10 == 0:
                    print(f"epoch: {epoch}/{self.num_epochs} "
                          f"batch: {batch_id}/{min(len(dataloader1), len(dataloader2))} "
                          f"step: {step} "
                          f"disc loss A: {discA_loss:.4f} "
                          f"disc loss B: {discB_loss:.4f} "
                          f"gen loss A2B: {genA2B_loss:.4f} "
                          f"gen loss B2A: {genB2A_loss:.4f}")

                    with torch.no_grad():
                        # plot generated and real images
                        img_grid_realA = torchvision.utils.make_grid(A_real[:16], normalize=True)
                        img_grid_fakeA = torchvision.utils.make_grid(A_fake[:16], normalize=True)

                        img_grid_realB = torchvision.utils.make_grid(B_real[:16], normalize=True)
                        img_grid_fakeB = torchvision.utils.make_grid(B_fake[:16], normalize=True)

                        writer_realA.add_image("Ground Truth A", img_grid_realA, global_step=step)
                        writer_fakeA.add_image("Generated A", img_grid_fakeA, global_step=step)

                        writer_realB.add_image("Ground Truth B", img_grid_realB, global_step=step)
                        writer_fakeB.add_image("Generated B", img_grid_fakeB, global_step=step)

                        writer_disc_lossG.add_scalar("disc/lossDiscB", discB_loss, global_step=step)
                        writer_gen_lossG.add_scalar("gen/lossGenA2B", genA2B_loss, global_step=step)

                        writer_disc_lossF.add_scalar("disc/lossDiscA", discA_loss, global_step=step)
                        writer_gen_lossF.add_scalar("gen/lossGenB2A", genB2A_loss, global_step=step)
                step += 1
