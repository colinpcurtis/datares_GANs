import torch.nn as nn
import torch.nn.functional as F


class cycleDiscriminator(nn.Module):
    def __init__(self, channels_img):
        super(cycleDiscriminator, self).__init__()
        self.channels_img = channels_img
        model = [nn.Conv2d(self.channels_img, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, image):
        x = self.model(image)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
