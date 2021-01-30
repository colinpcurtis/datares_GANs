import torch.nn as nn
from torch import cat


class conditionalDiscriminator(nn.Module):
    def __init__(self, channels_img):
        super().__init__()
        self.channels_img = channels_img
        self.conv1 = self._block(in_channels=self.channels_img*2, out_channels=64, kernel=3, stride=2)
        self.conv2 = self._block(in_channels=64, out_channels=128, kernel=3, stride=2)
        self.conv3 = self._block(in_channels=128, out_channels=256, kernel=3, stride=2)
        self.conv4 = self._block(in_channels=256, out_channels=512, kernel=3, stride=2)
        self.zeroPad = nn.ZeroPad2d(2)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1)
        self.BatchNorm = nn.BatchNorm2d(num_features=512)
        self.LeakyRelu = nn.LeakyReLU()
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1)
        self.Sigmoid = nn.Sigmoid()

    def _block(self, in_channels, out_channels, kernel, stride, batchnorm=True):
        # conv -> batch norm -> leaky relu
        # tensor (BatchSize, in_channels, img_height, img_width) -> (BatchSize, out_channels, img_height, img_width)
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel, stride=stride, bias=False))
        if batchnorm:
            block.add_module("batchnorm", nn.BatchNorm2d(out_channels))
        block.add_module("leakyRelu", nn.LeakyReLU())
        return block

    def forward(self, real, fake):
        # inputs are [BS, channels, 511, 511]
        # concatenate real and fake images on channels axis
        # see tensorflow link for detailed architecture
        concat = cat((real, fake), 1)
        # [BS, 2*channels, 511, 511]

        seq1 = self.conv1(concat)
        # [BS, 64, 255, 255]

        seq2 = self.conv2(seq1)
        # [BS, 128, 127, 127]

        seq3 = self.conv3(seq2)
        # [BS, 256, 63, 63]

        seq4 = self.conv4(seq3)
        # [BS, 512, 31, 31]

        seq5 = self.zeroPad(seq4)
        # [BS, 512, 35, 35]

        seq6 = self.conv5(seq5)
        # [BS, 512, 30, 30]

        seq7 = self.BatchNorm(seq6)

        seq8 = self.LeakyRelu(seq7)

        seq9 = self.zeroPad(seq8)

        seq10 = self.conv6(seq9)
        # [BS, 1, 34, 34]

        output = self.Sigmoid(seq10)
        return output
