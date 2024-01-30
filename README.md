# JSAC

## Installation
<details>
<summary><h3>Orin (from scratch)</h3></summary>
  
#### <ins>Finish setup</ins> 
```sudo apt update
sudo apt dist-upgrade
sudo reboot
sudo apt autoremove
sudo apt install nvidia-jetpack
```

Install the required packages
```
sudo apt -y install python3-pip
pip install --upgrade testresources setuptools wheel  
sudo apt-get -y install autoconf bc build-essential g++-10 gcc-10 clang-8 python3.8-dev
pip install numpy onnx --force-reinstall
```  

  
#### <ins>Install Jax</ins>  

**Option 1: Build Jax from Scratch (v0.4.12)**  
```
git clone -b jax-v0.4.12 https://github.com/google/jax
cd jax
python3 build/build.py --enable_cuda --cuda_compute_capabilities=sm_87
## Install the built jaxlib wheel
pip install -e .
```

**Option 2: Build Jax using wheel (v0.4.12, Orin JetPack 5.1.2)**  
Download the Jaxlib wheel from [here](https://drive.google.com/file/d/1UBxzqAxperW-4m44G1htKkAVEA6LfZlT/view?usp=drive_link) and install Jaxlib.  
Install Jax using ``` pip install jax==0.4.12 ```  
  
  
#### <ins>Update gcc, g++, and install the remaining libraries</ins>
```
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 1
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 2
sudo update-alternatives --config gcc

sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 1
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 2
sudo update-alternatives --config g++
```

```
pip install flax==0.7.2
pip install optax==0.1.7
pip install gym==0.23.1 matplotlib==3.7.2 tensorboardX==2.6.2 termcolor==2.3.0
pip install mujoco-py wandb seaborn pandas==1.5.3
```

</details>
<details>
  <summary><h3>Install Mujoco</h3></summary>

Download and save the _.mujoco_ folder at the _home_ directory ([Orin](https://drive.google.com/file/d/1B76qfcDcfFcU2Zc_LTSeeOt9JvtRvIwg/view?usp=drive_link), 
[Server](https://drive.google.com/file/d/1fmlISGPN6bvsDTYzDT-Q7mQ0Sx7I3zFo/view?usp=drive_link)).  

**Edit ~/.bashrc** to include Mujoco in $LD_LIBRARY_PATH:  
```export LD_LIBRARY_PATH=/home/{USERNAME}/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH```,  
and add the LD_PRELOAD variable:  
```export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libglapi.so.0```  

Install the required libraries: 
```
pip install Cython==0.29.36 
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
sudo apt-get install patchelf
```
</details>

### Server
Create a conda environment and install the required libraries.  
```
conda create -n jsac python=3.8 jaxlib=*0.4.12=*cuda* jax==0.4.12 cuda-nvcc -c conda-forge -c nvidia
conda activate jsac

conda install pip

pip install opencv-python flax==0.7.2 gym==0.23.1 matplotlib==3.7.2 optax==0.1.5
pip install tensorboardX==2.6.2 termcolor==2.3.0 mujoco-py wandb seaborn pandas==1.5.3

```
<br>  

### Training

**Edit ~/.bashrc** to include the path to the JSAC folder in $PYTHONPATH:  
```export PYTHONPATH=/home/{PATH_TO_JSAC}/JSAC:$PYTHONPATH```  
<br>
Run a [task](https://github.com/fahimfss/JSAC/tree/master/tasks/simulation) file using:  
```python3 task_mujoco_sync_img_prop.py --seed 41```  


