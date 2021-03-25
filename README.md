# DataRes GANs


### Installation and Setup

It is recommended to run this project on a cloud instance to make easy use of GPU computing.  Once the VM is set up, clone
the repo with ```git clone https://github.com/colinpcurtis/datares_GANs.git```.  Then ```cd datares_GANs``` , 
and run ```./install.sh```to install gcc, the NVIDIA drivers, and CUDA.  You will have to manually agree 
to a few commands.  

Note: The shell script only works on Ubuntu 18.04, so make sure to select this Linux OS while
configuring the VM.

If that installation is successful, the nvidia control panel should be displayed on the window.  

To install python requirements, run ```pip install -r requirements.txt```.  

To test if everything is working correctly, open the python shell and run the following command, which should return
true.
```python
>> import torch
>> torch.cuda.is_available()
True
```

### Running Deep Convolutional GAN
To train the model, run 
```python3 run.py -m DCGAN -e 10 -l logs```
This command runs the DCGAN model with the MNIST dataset for 10 epochs, 
and saves tensorboard results to the ```logs``` directory.  
The TensorBoard results can be accessed by running ```tensorboard --logdir ./logs```.

It is recommended to download the logs file locally and then run TensorBoard so as not to waste expensive GPU time.

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

It takes approximately 1.5 hours to train the Cycle GAN for 10 epochs using float 32.  

Google Cloud SDK (if using a gcloud VM) must be installed for port forwarding.  
