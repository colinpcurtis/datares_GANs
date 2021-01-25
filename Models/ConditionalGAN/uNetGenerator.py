import torch.nn as nn
from torch import cat


class conditionalGenerator(nn.Module):
    def __init__(self, channels_img):
        super().__init__()
        self.channels_img = channels_img
        self.filters = 16

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
        # upsampling layers generate encode spatially of features  and downsampling capures the features

        # print("0: ", x.size())
        # [BS, 1, 511, 511]
        # [BS, channels, img_height, img_width]

        down1 = self._block_downsample(in_channels=self.channels_img, out_channels=self.filters*4,
                                       kernel=3, stride=2)(x)
        # [BS, 64, 255, 255]
        # print("1: ", down1.size())

        down2 = self._block_downsample(in_channels=self.filters*4, out_channels=self.filters*8,
                                       kernel=3, stride=2)(down1)
        # [BS, 64, 127, 127]
        # print("2: ", down2.size())

        down3 = self._block_downsample(in_channels=self.filters*8, out_channels=self.filters*16,
                                       kernel=3, stride=2)(down2)
        # [BS, 128, 63, 63]
        # print("3: ", down3.size())

        down4 = self._block_downsample(in_channels=self.filters*16, out_channels=self.filters*32,
                                       kernel=3, stride=2)(down3)
        # [BS, 256, 31, 31]
        # print("4: ", down4.size())

        down5 = self._block_downsample(in_channels=self.filters*32, out_channels=self.filters*32,
                                       kernel=3, stride=2)(down4)
        # [BS, 512, 15, 15]
        # print("5: ", down5.size())

        down6 = self._block_downsample(in_channels=self.filters*32, out_channels=self.filters*32,
                                       kernel=3, stride=2)(down5)
        # [BS, 512, 7, 7]
        # print("6: ", down6.size())

        down7 = self._block_downsample(in_channels=self.filters*32, out_channels=self.filters*32,
                                       kernel=3, stride=2)(down6)
        # [BS, 512, 3, 3]
        # print("7: ", down7.size())

        down8 = self._block_downsample(in_channels=self.filters*32, out_channels=self.filters*32,
                                       kernel=3, stride=2, batchnorm=False)(down7)
        # [BS, 512, 1, 1]
        # need batchnorm=false since it doesn't expect a 1x1 image
        # print("bottom: ", down8.size())

        up7 = self._block_upsample(in_channels=self.filters*32, out_channels=self.filters*32,
                                   kernel=3, stride=2)(down8)
        # [BS, 512, 3, 3]
        # print("7: ", up7.size())

        concat1 = cat((up7, down7), 1)
        # [BS, 1024, 3, 3]
        # concatenate then downsample to half size
        # print("concat1 ", concat1.size())

        seq1 = self._block_upsample(in_channels=self.filters*64, out_channels=self.filters*32,
                                    kernel=3, stride=2)(concat1)
        # [BS, 512, 7, 7]
        # print("6: ", seq1.size())

        concat2 = cat((seq1, down6), 1)
        # [BS, 1024, 7, 7]
        # print("concat2 ", concat2.size())

        seq2 = self._block_upsample(in_channels=self.filters*64, out_channels=self.filters*32,
                                    kernel=3, stride=2)(concat2)
        # [BS, 512, 15, 15]
        # print("5:  ", seq2.size())

        concat3 = cat((seq2, down5), 1)
        # [BS, 1024, 15, 15]
        # print("concat3", concat3.size())

        seq3 = self._block_upsample(in_channels=self.filters*64, out_channels=self.filters*32,
                                    kernel=3, stride=2)(concat3)
        # [BS, 512, 31, 31]
        # print("4: ", seq3.size())

        concat4 = cat((seq3, down4), 1)
        # [BS, 1024, 31, 31]
        # print("concat4 ", concat4.size())

        seq4 = self._block_upsample(in_channels=self.filters*64, out_channels=self.filters*16,
                                    kernel=3, stride=2)(concat4)
        # [BS, 256, 63, 63]
        # print("3:  ", seq4.size())

        concat5 = cat((seq4, down3), 1)
        # [BS, 512, 63, 63]
        # print("concat5 ", concat5.size())

        seq5 = self._block_upsample(in_channels=self.filters*32, out_channels=self.filters*8,
                                    kernel=3, stride=2)(concat5)
        # [BS, 128, 127, 127]
        # print("2:  ", seq5.size())

        concat6 = cat((seq5, down2), 1)
        # [BS, 256, 127, 127]
        # print("concat6 ", concat6.size())

        seq6 = self._block_upsample(in_channels=self.filters*16, out_channels=self.filters*4,
                                    kernel=3, stride=2)(concat6)
        # [BS, 64, 255, 255]
        # print("1:  ", seq6.size())

        concat7 = cat((seq6, down1), 1)
        # [BS, 64, 255, 255]
        # print("concat7 ", concat7.size())

        # TODO: should do normal conv transposed here then add sigmoid activation

        seq7 = self._block_upsample(in_channels=self.filters*8, out_channels=self.channels_img,
                                    kernel=3, stride=2)(concat7)
        # [BS, 1, 511, 511]
        # print("0: ", seq7.size())
        # return to original image shape
        return seq7

    # def gen_loss(self, disc_generated_output, output, target, lambda_penalty):
    #     # computes cross entropy between discriminator output and ideal output
    #     # also uses L1 loss to encourage feature matching in generator
    #     loss = nn.BCELoss(ones_like(disc_generated_output), output)
    #
    #     l1_loss = nn.L1Loss(abs(output - target))
    #     total_loss = loss + l1_loss


