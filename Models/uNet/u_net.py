import torch.nn as nn
from torch import cat

class UNet(nn.Module):
    def __init__(self, channels_img):
        super().__init__()
        self.channels_img = channels_img
        self.filters = 16

        self.net = nn.Sequential()

    def _block_downsample(self, in_channels, out_channels, kernel, stride):
        # tensor (BatchSize, in_channels, img_height, img_width) -> (BatchSize, out_channels, img_height, img_width)
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )
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

        # this is basically the whole u-net network but it will just contain more layers and up-sampline layers
        # have to be concatenated with corresponding down-sampling layers

        print("input: ", x.size())
        # run forward pass on tensor x of shape (BS, channels_img, img_height, img_width)

        down1 = self._block_downsample(in_channels=self.channels_img, out_channels=self.filters, kernel=3, stride=2)(x)
        print("down 1: ", down1.size())

        down2 = self._block_downsample(in_channels=self.filters, out_channels=self.filters*2, kernel=3, stride=2)(down1)
        print("down 2: ", down2.size())

        down3 = self._block_downsample(in_channels=self.filters*2, out_channels=self.filters*4, kernel=3, stride=2)(down2)
        print("down 3: ", down3.size())

        down4 = self._block_downsample(in_channels=self.filters*4, out_channels=self.filters*8, kernel=3, stride=2)(down3)
        print("down 4: ", down4.size())

        up4 = self._block_upsample(in_channels=self.filters*8, out_channels=self.filters*4, kernel=3, stride=2)(down4)
        print("up 4: ", up4.size())

        up3 = self._block_upsample(in_channels=self.filters*4, out_channels=self.filters*2, kernel=3, stride=2)(up4)
        print("up 3: ", up3.size())

        up2 = self._block_upsample(in_channels=self.filters*2, out_channels=self.filters, kernel=3, stride=2)(up3)
        print("up 2: ", up2.size())

        up1 = self._block_upsample(in_channels=self.filters, out_channels=self.channels_img, kernel=3, stride=2)(up2)
        print("up 1: ", up1.size())

        # concat along first axis, since zeroth axis is for batch size
        # up1 = cat((x, up1), 1)
        # print(up1.size())

        return down1
