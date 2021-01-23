from u_net import UNet
import torch


channels_img = 1
net = UNet(channels_img=channels_img)

# since down-sampling uses the floor we sometimes get weird rounding, so you have to be a bit careful
# to make sure that image sizes on encoder and decoder are equal in corresponding layers
img_size = 63
fixed_noise = torch.randn(1, channels_img, img_size, img_size)

# test to make sure sizes are as expected while developing
net(fixed_noise)
