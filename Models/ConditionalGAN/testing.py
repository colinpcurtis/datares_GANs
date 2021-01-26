from uNetGenerator import conditionalGenerator
from uNetDiscriminator import conditionalDiscriminator
import torch

# dataset link: https://nihcc.app.box.com/v/DeepLesion/folder/50715173939
# reference for architecture: https://www.tensorflow.org/tutorials/generative/pix2pix

channels_img = 1
gen = conditionalGenerator(channels_img=channels_img)
disc = conditionalDiscriminator(channels_img=channels_img)

# since down-sampling uses the floor we sometimes get weird rounding, so you have to be a bit careful
# to make sure that image sizes on encoder and decoder are equal in corresponding layers
img_size = 511
start_image = torch.randn(1, channels_img, img_size, img_size)

# test to make sure sizes are as expected while developing
# channels should double during downsampling and be halved during upsampling
fake = gen(start_image)
print("generator output: ", fake.size())
print("")
output = disc(start_image, fake)
print("discriminator output: ", output.size())
print(output.mean().item())
