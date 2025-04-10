#!/bin/bash
#SBATCH --job-name=run_lang_splat_debug
#SBATCH --output=sbatch_log/run_lang_splat_debug_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

#SBATCH --nodelist=bmicgpu07,bmicgpu08,bmicgpu09,bmicgpu10,octopus01,octopus02,octopus03,octopus04
#SBATCH --cpus-per-task=4
#SBATCH --mem 128GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qi.ma@vision.ee.ethz.ch


source /scratch_net/schusch/qimaqi/miniconda3/etc/profile.d/conda.sh
conda activate langsplat_cu116
export CC=/scratch_net/schusch/qimaqi/install_gcc_8_5/bin/gcc-8.5.0
export CXX=/scratch_net/schusch/qimaqi/install_gcc_8_5/bin/g++-8.5.0
# export CONDA_OVERRIDE_CUDA=11.8
# export CUDA_HOME=$CONDA_PREFIX
# export PATH=$CUDA_HOME/bin:$PATH

cd /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/LangSplat_Qi/

python langsplat_jobs.py --start_idx=0 --end_idx=1 