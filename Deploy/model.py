import torch
import io
import torchvision.transforms as transforms
from PIL import Image
import os, sys
from setup import IMAGE_SIZE 

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


def transform(image_bytes):
    # apply specified transforms on the image and return transformed image
    # we don't reapply Gaussian Noise because that's only for training
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
    # open image at path and convert to RGB, then save at same location
    # need this because some image formats are not by default 3 channels
    # for example .png sometimes is 4 color channels and this would break
    # the server
    Image.open(path).convert('RGB').save(path)
    with open(path, "rb") as f:
        image_bytes = f.read()

    return image_bytes


def get_prediction(model, im_path: str):
    im_bytes = load_im(im_path)

    tensor = transform(im_bytes)

    # do foward pass on model to get prediction
    prediction = model.forward(tensor).squeeze()  # squeeze makes image [3, 512, 512]
    assert prediction.size() == torch.Size([3, 512, 512])

    tensorToPIL = transforms.ToPILImage(mode="RGB")
    # need to un-normalize the image using mean and std
    z = prediction * torch.full_like(prediction, .5)  
    z = z + torch.full_like(z, .5)

    return tensorToPIL(z)
