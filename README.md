# JSAC

## Installation

### Orin (From Scratch)
#### Finish setup 
```sudo apt update
sudo apt dist-upgrade
sudo reboot
sudo apt autoremove
sudo apt install nvidia-jetpack
```

Install the required packages
```
sudo apt -y install python3-pip
pip3 install --upgrade testresources setuptools wheel  
sudo apt-get -y install autoconf bc build-essential g++-10 gcc-10 clang-8 python3.8-dev
pip3 install numpy onnx --force-reinstall
```

#### Install Jax  

**Option 1: Build Jax from Scratch (v0.4.12)**  
``` git clone -b jax-v0.4.12 https://github.com/google/jax
cd jax
python3 build/build.py --enable_cuda --cuda_compute_capabilities=sm_87
## Install the built jaxlib wheel
pip3 install -e .
```

**Option 2: Build Jax using wheel (v0.4.12, Orin JetPack 5.1.2)**  
Download the Jaxlib wheel from [here](https://drive.google.com/drive/folders/1d1ZoQsR65EWWjmUa_K6ib4bk8AlyxEls?usp=drive_link) and install Jaxlib.  
Install Jax using ``` pip3 install jax==0.4.12 ```


#### Install Flax
**Make gcc-10 default gcc and g++-10 default g++**
```
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 1
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 2
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 3
sudo update-alternatives --config gcc

sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 1
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 2
sudo update-alternatives --config g++
```

**Install Flax and Optax**
```
pip3 install flax==0.7.0
pip3 install optax==0.1.5
```

#### Install Mujoco 
Download [.mujoco_orin.zip](https://drive.google.com/drive/folders/1d1ZoQsR65EWWjmUa_K6ib4bk8AlyxEls?usp=drive_link), unzip and save .mujoco folder at the _home_ directory.
Edit ~/.bashrc to include Mujoco in $LD_LIBRARY_PATH. ([Reference](https://drive.google.com/file/d/1cZRVREH0HuVIQDLhRcKHvPAjzXw6y5ZR/view?usp=drive_link)).  
Install the required libraries: 
```
pip3 install Cython==0.29.36 
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
sudo apt-get install patchelf
```

### Workstation
Create a conda environment and install the required libraries.  
```
conda create -n jsac python=3.8 jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
conda activate jsac

conda install pip

pip3 install opencv-python flax==0.7.0 gym==0.23.1 matplotlib==3.7.2 optax==0.1.5 tensorboardX==2.6.2 termcolor==2.3.0 mujoco-py wandb 

```
