import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]
        # nn.ReflectionPad2d(1), nn.ReflectionPad2d(1),
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class cycleGenerator(nn.Module):
    def __init__(self, image_size, n_residual_blocks=9):
        super(cycleGenerator, self).__init__()

        # Initial convolution block nn.ReflectionPad2d(3),
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(image_size, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer nn.ReflectionPad2d(3),
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, image_size, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    import torch
    gen = cycleGenerator(1)
    x = torch.rand(1, 1, 512, 512)
    x = gen.forward(x)
    print(x.size())

    y = torch.mean((x - torch.ones_like(x)))
    print(y)
