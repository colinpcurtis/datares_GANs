import torch.nn as nn
from torch import cat


class conditionalGenerator(nn.Module):
    def __init__(self, channels_img):
        super().__init__()
        self.channels_img = channels_img
        self.filters = 16
        self.conv1 = self._block_downsample(in_channels=self.channels_img, out_channels=self.filters * 4,
                                            kernel=3, stride=2)
        self.conv2 = self._block_downsample(in_channels=self.filters * 4, out_channels=self.filters * 8,
                                            kernel=3, stride=2)
        self.conv3 = self._block_downsample(in_channels=self.filters * 8, out_channels=self.filters * 16,
                                            kernel=3, stride=2)
        self.conv4 = self._block_downsample(in_channels=self.filters * 16, out_channels=self.filters * 32,
                                            kernel=3, stride=2)
        self.conv5 = self._block_downsample(in_channels=self.filters * 32, out_channels=self.filters * 32,
                                            kernel=3, stride=2)
        self.conv6 = self._block_downsample(in_channels=self.filters * 32, out_channels=self.filters * 32,
                                            kernel=3, stride=2)
        self.conv7 = self._block_downsample(in_channels=self.filters * 32, out_channels=self.filters * 32,
                                            kernel=3, stride=2)
        self.conv8 = self._block_downsample(in_channels=self.filters * 32, out_channels=self.filters * 32,
                                            kernel=3, stride=2, batchnorm=False)
        self.conv9 = self._block_upsample(in_channels=self.filters * 32, out_channels=self.filters * 32,
                                          kernel=3, stride=2)
        self.conv10 = self._block_upsample(in_channels=self.filters * 64, out_channels=self.filters * 32,
                                           kernel=3, stride=2)
        self.conv11 = self._block_upsample(in_channels=self.filters * 64, out_channels=self.filters * 32,
                                           kernel=3, stride=2)
        self.conv12 = self._block_upsample(in_channels=self.filters * 64, out_channels=self.filters * 32,
                                           kernel=3, stride=2)
        self.conv13 = self._block_upsample(in_channels=self.filters * 64, out_channels=self.filters * 16,
                                           kernel=3, stride=2)
        self.conv14 = self._block_upsample(in_channels=self.filters * 32, out_channels=self.filters * 8,
                                           kernel=3, stride=2)
        self.conv15 = self._block_upsample(in_channels=self.filters * 16, out_channels=self.filters * 4,
                                           kernel=3, stride=2)
        self.conv16 = nn.ConvTranspose2d(in_channels=self.filters * 8, out_channels=self.channels_img,
                                         kernel_size=3, stride=2)

    def _block_downsample(self, in_channels, out_channels, kernel, stride, batchnorm=True):
        # tensor (BatchSize, in_channels, img_height, img_width) -> (BatchSize, out_channels, img_height, img_width)
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel, stride=stride, bias=False))
        if batchnorm:
            block.add_module("batchnorm", nn.BatchNorm2d(out_channels))
        block.add_module("leakyRelu", nn.LeakyReLU())
        # TODO: check if bias should be true or false
        return block

    def _block_upsample(self, in_channels, out_channels, kernel, stride):
        # tensor (BatchSize, in_channels, img_height, img_width) -> (BatchSize, out_channels, img_height, img_width)
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        return block

    def forward(self, x):
        # idea - concatenate along channels axis the corresponding upsampling and downsampling layers
        # upsampling layers generate encode spatially of features and downsampling captures the features

        # [BS, 1, 511, 511]
        # [BS, channels, img_height, img_width]

        down1 = self.conv1(x)
        # [BS, 64, 255, 255]

        down2 = self.conv2(down1)
        # [BS, 64, 127, 127]

        down3 = self.conv3(down2)
        # [BS, 128, 63, 63]

        down4 = self.conv4(down3)
        # [BS, 256, 31, 31]

        down5 = self.conv5(down4)
        # [BS, 512, 15, 15]

        down6 = self.conv6(down5)
        # [BS, 512, 7, 7]

        down7 = self.conv7(down6)
        # [BS, 512, 3, 3]

        down8 = self.conv8(down7)
        # [BS, 512, 1, 1]

        up7 = self.conv9(down8)
        # [BS, 512, 3, 3]

        concat1 = cat((up7, down7), 1)
        # [BS, 1024, 3, 3]
        # concatenate then downsample to half size

        seq1 = self.conv10(concat1)
        # [BS, 512, 7, 7]

        concat2 = cat((seq1, down6), 1)
        # [BS, 1024, 7, 7]

        seq2 = self.conv11(concat2)
        # [BS, 512, 15, 15]

        concat3 = cat((seq2, down5), 1)
        # [BS, 1024, 15, 15]

        seq3 = self.conv12(concat3)
        # [BS, 512, 31, 31]

        concat4 = cat((seq3, down4), 1)
        # [BS, 1024, 31, 31]

        seq4 = self.conv13(concat4)
        # [BS, 256, 63, 63]

        concat5 = cat((seq4, down3), 1)
        # [BS, 512, 63, 63]

        seq5 = self.conv14(concat5)
        # [BS, 128, 127, 127]

        concat6 = cat((seq5, down2), 1)
        # [BS, 256, 127, 127]

        seq6 = self.conv15(concat6)
        # [BS, 64, 255, 255]

        concat7 = cat((seq6, down1), 1)
        # [BS, 64, 255, 255]

        seq7 = self.conv16(concat7)
        # [BS, 1, 511, 511]
        # returns to original image shape
        return seq7
