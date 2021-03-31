# DataRes GANs

### Installation and Setup

It is recommended to run this project on a Google Cloud VM instance preinstalled with NVIDIA drivers, CUDA, and pytorch.  
Once the VM is set up, clone the repo with ```git clone https://github.com/colinpcurtis/datares_GANs.git``` 
and ```cd datares_GANs```.

### Running the Conditional and Cycle GAN

#### Fetching the dataset
Run ```python3 fetch_dataset.py```.  This downloads and unzips the dataset zip files into the ```/datasets``` directory.   
Then run ```./move.sh``` to move the dataset into a format so that the PyTorch dataset class can read 
the images.

#### Training
There are numerous command line arguments to run the model.  For simplicity purposes they can be found and changed in 
the ```Makefile```.  To run the model, simply invoke the ```make``` command and training will start.  

In a new terminal window, run ```tensorboard --logdir ./logs``` to open the TensorBoard logs.  This has to be 
port forwarded if in a virtual machine, so run ```gcloud compute ssh [INSTANCE_NAME] -- -NfL 6006:localhost:6006```
in the local terminal to access the logs. Then visiting ```http://localhost:6006``` in a web browser 
will display the training logs.

It takes approximately 3 hours to train the Cycle GAN for 10 epochs using 5 residual layers and float 32.  

Google Cloud SDK (if using a gcloud VM) must be installed for port forwarding.  
