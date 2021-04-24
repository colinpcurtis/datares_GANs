# DataRes GANs

### Installation and Setup

It is recommended to run this project on a _Deep Learning with Linux_ Google Cloud VM instance preinstalled with NVIDIA drivers, CUDA 11, and PyTorch 1.8.   
Once the VM is set up, clone the repo with ```git clone https://github.com/colinpcurtis/datares_GANs.git``` 
and ```cd datares_GANs```.

### Running the Conditional and Cycle GAN

#### Fetching the dataset
Run ```make dataset```.  This downloads and unzips the dataset zip files into the ```/datasets``` directory.
Then run ```./move.sh``` to move the dataset into a format so that the PyTorch dataset class can read 
the images.

#### Training
There are numerous command line arguments to run the model.  For simplicity purposes they can be found and changed in 
the ```Makefile```.  To run the model, simply invoke the ```make``` command and training will start.

There may be some _CUDA out of memory errors_ depending on the GPU that is used for training.  In that case, change the _BATCH\_SIZE_ variable to something smaller in ```Models/cycleGAN/CycleGAN.py```

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

Then a generated image from the Cycle GAN can be seen below.

![Generated Image](/Deploy/pred3.jpg)
