"""
    Defines helper functions for reloading our model
    and transforming the loaded images

"""
import torch
import io
import torchvision.transforms as transforms
from PIL import Image

from Models.cycleGAN.CycleGenerator import CycleGenerator
from Models.cycleGAN.CycleGAN import IMAGE_SIZE
from config import PROJECT_ROOT

def transform(im_bytes):
    # apply specified transforms on the image and return transformed image
    transform_list = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                            # transforms.CenterCrop(IMAGE_SIZE),
                                            transforms.ToTensor(),
                                            transforms.Normalize(.5, .5)])
    image = Image.open(io.BytesIO(image_bytes))
    return transform_list(image).unsqueeze(0)

def load_model(path: str):
    model = CycleGenerator(IMAGE_SIZE)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def get_prediction(model, im_bytes):
    tensor = transform(im_bytes)
    prediction = model.forward(tensor)
    return prediction

if __name__ == "__main__":
    # gen = load_model(f"{PROJECT_ROOT}/TrainedModels/genB2A")
    with open(f"test_im.png", "rb") as f:
        image_bytes = f.read()
        tensor = transform(image_bytes)
        print(tensor)

