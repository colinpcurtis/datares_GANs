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

### Running the model
To train the model, run 
```python
python3 run.py -m DCGAN -e 10 -v 1 -s logs
```
This command runs the DCGAN model for 10 epochs, and saves tensorboard results to the ```logs``` directory.  The
TensorBoard results can be accessed by running ```tensorboard --logdir ./logs```.

It is recommended to download the logs file locally and then run TensorBoard so as not to waste expensive GPU time.