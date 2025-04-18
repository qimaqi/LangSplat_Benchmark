#!/bin/bash
#SBATCH --job-name=nnunet_baseline
#SBATCH --output=sbatch_log/convformer0_2layer_acdc_inception_conv_elu_debug_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

#SBATCH --nodelist=octopus04
#SBATCH --cpus-per-task=4
#SBATCH --mem 120GB



# cuda13

# cuda14
source /scratch_net/schusch/qimaqi/miniconda3/etc/profile.d/conda.sh
conda create -n langsplat_cu14 python=3.7 -y
conda activate langsplat_cu14
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

pip install numpy tqdm opencv-python scikit-image scikit-learn matplotlib tensorboardX plyfile  open-clip-torch
export CC=/scratch_net/schusch/qimaqi/install_gcc_8_5/bin/gcc-8.5.0
export CXX=/scratch_net/schusch/qimaqi/install_gcc_8_5/bin/g++-8.5.0



# cuda117

# install 
conda create -n langsplat_cu18 python=3.11 -y

conda activate langsplat_cu18

# export CONDA_OVERRIDE_CUDA=11.8
# export CUDA_HOME=$CONDA_PREFIX
# export PATH=$CUDA_HOME/bin:$PATH

conda install -c nvidia/label/cuda-11.7.0 cuda-toolkit=11.7 cuda-nvcc=11.7 -y 
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install numpy==1.24.1 tqdm opencv-python scikit-image scikit-learn matplotlib tensorboardX plyfile  open-clip-torch



conda create -n langsplat_cu18 python=3.9 -y

conda activate langsplat_cu18

export CONDA_OVERRIDE_CUDA=11.8
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH

conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit=11.8 cuda-nvcc=11.8 -y 
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.8 -c pytorch -c conda-forge
pip install numpy==1.24.1 tqdm opencv-python scikit-image scikit-learn matplotlib tensorboardX plyfile 


conda create -n langsplat_cu18_test python=3.9 -y

conda activate langsplat_cu18_test

export CONDA_OVERRIDE_CUDA=11.8
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH

conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit=11.8 cuda-nvcc=11.8 -y 
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
# pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.1_cu118.html
pip install numpy==1.24.1 tqdm opencv-python scikit-image scikit-learn matplotlib tensorboardX plyfile open-clip-torch numba open3d




cmake .. -DOpenCV_DIR=$CONDA_PREFIX/lib/cmake/opencv4

conda install -c conda-forge opencv
conda install -c conda-forge cmake==3.16

# cmake -DOpenCV_DIR=/scratch_net/schusch/qimaqi/miniconda3/envs/renderpy/cmake/opencv4 ..
# /home/qi/miniconda3/envs/your_env/lib/cmake/opencv4