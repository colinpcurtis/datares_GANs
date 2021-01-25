from torch import nn
from torch import device
from torch.cuda import is_available
from torch.optim import Adam
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
LEARNING_RATE = 1e-4
BATCH_SIZE = 256  # optimize to max out batch size while training
IMAGE_SIZE = 511
CHANNELS_IMG = 3


# class subtract(object):
#     """
#         params: 8 bit PIL image
#         returns: image subtracted by large number to get grayscale
#     """
#     def __call__(self, img):
#         """
#             Args:
#                 PIL image
#             returns:
#                 PIL grayscale subtracted by 32768
#         """
#         img.
#         # data = asarray(img)
#         # # print(data)
#         # data = data - 32768
#
#         new_img = Image.fromarray(np.uint8(data)).convert('RGB')
#         # new_img.show()
#         return new_img


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
    def __init__(self, num_epochs, save_dir):
        self.num_epochs = num_epochs
        self.save_dir = save_dir

    def transform(self):
        data_transforms = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor(),
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

        gen = conditionalGenerator(channels_img=CHANNELS_IMG)
        disc = conditionalDiscriminator(channels_img=CHANNELS_IMG)

        loss = nn.BCELoss()

        # train loop
        for epoch in range(self.num_epochs):
            for batch_id, (real, _) in enumerate(dataloader):
                fake = gen(real)
                print("size: ", fake.size())
                quit()


gan = conditionalGAN(num_epochs=1, save_dir="test")
gan.train()
