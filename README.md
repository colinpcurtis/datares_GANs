# UCLA DataRes Spring 2021: Make-A-Monet

## Project Goal

The goal of this repository is to build and deploy a Cycle GAN for style transfer of Monet paintings.  Although there is significant work on GAN training, we have found there is not much work on deploying them to the inference step given their difficulty to train.  

In this case we build and train our own Cycle GAN using the [monet2photo](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/) dataset.  Then once it is trained we deploy the photo to painting generator to a Heroku webpage with ```int8``` quantization to maximize performance and RAM usage on the small deployment machine.  

## Installation and Setup

It is recommended to run this project on a _Deep Learning with Linux_ Google Cloud VM instance preinstalled with NVIDIA drivers, CUDA 11, and PyTorch 1.8.   
Once the VM is set up, clone the repo with ```git clone https://github.com/colinpcurtis/datares_GANs.git``` 
and ```cd datares_GANs```.

## Running the Conditional and Cycle GAN

### Fetching the dataset
Run ```make dataset```.  This downloads and unzips the dataset zip files into the ```/datasets``` directory.
Then run ```./move.sh``` to move the dataset into a format so that the PyTorch dataset class can read 
the images.

### Training
There are numerous command line arguments to run the model.  For simplicity purposes they can be found and changed in 
the ```Makefile```.  To run the model, simply invoke the ```make``` command and training will start.

At the start of training there may be ```CUDA out of memory``` errors depending on the GPU using in the VM.  In that case, change the ```BATCH_SIZE``` variable to something smaller in ```Models/cycleGAN/CycleGAN.py```

In a new terminal window, run ```tensorboard --logdir ./logs``` to open the TensorBoard logs.  This has to be
port forwarded to a local computer, so run 
```gcloud compute ssh user@instance_name -- -NfL 6006:localhost:6006```
in the local terminal to access the logs. Then visiting ```http://localhost:6006``` in a web browser 
will display the training logs.

It takes approximately 3 hours to train the Cycle GAN for 10 epochs using 5 residual layers and float 32.  

Google Cloud SDK (if using a gcloud VM) must be installed for port forwarding.  


## Preliminary Results
The final iteration of the model will allow users to upload images onto a web app, but some initial results can be seen below.  

This large image of Royce Hall has to be cropped to a 512x512 size in order to fit in the model.
![Raw Image](/Deploy/test3.jpeg)

Then a generated image from the Cycle GAN can be seen below.  Seeing as the goal was to make Monet-style paintings from photos, we can definitely see some realistic results.

![Generated Image](/Deploy/pred3.jpg)


## Running the Deployment Webpage Locally

It is recommended to run the inference model as a Docker container, and as such Docker should be installed on the system.

To build the image, run ```docker build -t gans_deploy .```, which will take a few minutes to run.  

Then, run ```docker run -d -p 8888:8888 gans_deploy``` and visit ```http://localhost:8888``` to use the webpage.  