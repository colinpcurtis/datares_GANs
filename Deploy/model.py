"""
    Colin Curtis - 04/23/2021
    Defines helper functions for reloading our model
    and transforming the loaded images
"""

import torch
import io
import torchvision.transforms as transforms
from PIL import Image
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from Models.cycleGAN.CycleGAN import IMAGE_SIZE
from config import PROJECT_ROOT


def transform(image_bytes):
    # apply specified transforms on the image and return transformed image
    transform_list = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                         transforms.CenterCrop(IMAGE_SIZE),
                                         transforms.ToTensor(),
                                         transforms.Normalize(.5, .5)])
    image = Image.open(io.BytesIO(image_bytes))
    return transform_list(image).unsqueeze(0)


def load_model(path: str):
    model = torch.load(path, map_location='cpu')
    model.eval()
    return model


def load_im(path: str):
    Image.open(path).convert('RGB').save(path)
    with open(path, "rb") as f:
        image_bytes = f.read()

    return image_bytes


def get_prediction(model, im_path: str):
    # TODO: add asserts to make sure everything is of the correct size
    im_bytes = load_im(im_path)

    tensor = transform(im_bytes)

    # do foward pass on model to get prediction
    prediction = model.forward(tensor).squeeze()  # squeeze makes image [3, 512, 512]
    assert prediction.size() == torch.Size([3, 512, 512])

    tensorToPIL = transforms.ToPILImage(mode="RGB")

    z = prediction * torch.full_like(prediction, .5)  # un-normalize the image using mean and std
    z = z + torch.full_like(z, .5)

    return tensorToPIL(z)


if __name__ == "__main__":
    # simple script for testing

    gen = load_model(f"{PROJECT_ROOT}/TrainedModels/genB2A.pt")

    pred = get_prediction(gen, "test3.jpeg")
    pred.save("pred3.jpg")
