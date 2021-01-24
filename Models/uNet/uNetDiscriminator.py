import torch.nn as nn
from torch import cat


class UNetDiscriminator(nn.Module):
    def __init__(self, channels_img):
        super().__init__()
        self.channels_img = channels_img

    def _block(self):
        # conv -> batch norm -> leaky relu
        pass

    def forward(self, real, fake):
        # concatenate real and fake images on channels axis
        # see tensorflow link for detailed architecture
        pass
