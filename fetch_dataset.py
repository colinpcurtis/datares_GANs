import threading
import requests
import zipfile
import io
from config import PROJECT_ROOT

links = [
    "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/monet2photo.zip",
    "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/vangogh2photo.zip",
]


def download_file(link):
    r = requests.get(link)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(f"{PROJECT_ROOT}/datasets")


def download_thread(link):
    thread = threading.Thread(target=download_file, args=(link,))
    thread.start()


for idx, link in enumerate(links):
    print("downloading ", idx)
    download_thread(link)

print("Please wait for the function to return; it might take a few minutes")
