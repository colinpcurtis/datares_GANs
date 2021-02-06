#!/bin/sh
sudo apt-get update

# remove previous nvidia installations
sudo apt-get purge nvidia*
sudo apt-get autoremove
sudo apt-get autoclean
sudo rm -rf /usr/local/cuda*

# install gcc (c compiler needed for cuda)
sudo apt-get gcc

# install cuda 10 and drivers
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" | sudo tee /etc/apt/sources.list.d/cuda.list
sudo apt-get update
sudo apt-get -o Dpkg::Options::="--force-overwrite" install cuda-10-0 cuda-drivers

# reboot machine
sudo reboot

# update bashrc
echo 'export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc
sudo ldconfig

# install pip
echo "installing pip"
sudo apt install python3-pip

# nvidia control pannel should show up to work properly
nvidia-smi

echo "If you see nvidia control panel then the installation succeeded"
