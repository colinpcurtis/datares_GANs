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
The
TensorBoard results can be accessed by running ```tensorboard --logdir ./logs```.

It is recommended to download the logs file locally and then run TensorBoard so as not to waste expensive GPU time.

### Running the Conditional and Cycle GAN

#### Fetching the dataset 
Run ```python3 fetch_dataset.py```.  Only roughly a quarter of the entire dataset will be downloaded since the other 
links are commented out.  This will speed up the initial testing until a whole run is executed.  Then run 
```./unzip.sh``` to unzip the compressed files and remove the Zip files.

#### Training
The ```python3 run.py -m conditionalGAN -e 10 -l logs -s modelFile.pt -d monet2photo``` command will run the Conditional GAN for 10 epochs, 
save the results to the ```/logs``` directory, and save the model state dict as ```modelFile.pt``` in the project root
directory.  We specify to train the model on the ```monet2photo``` dataset directory.  

In a new terminal window, run ```tensorboard --logdir ./logs``` to open the TensorBoard logs.  This has to be 
port forwarded if in a virtual machine, so run ```gcloud compute ssh [INSTANCE_NAME] -- -NfL 6006:localhost:6006```
in the local terminal to access the logs. Then visiting ```http://localhost:6006``` in a web browser 
will display the training logs.  

Google Cloud SDK (if using a gcloud VM) must be installed for port forwarding.  
