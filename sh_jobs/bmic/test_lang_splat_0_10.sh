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


# python render.py -s dataset_path -m output/${casename} --feature_level ${level}

# python render.py -s /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/data/09c1414f1b -m output/09c1414f1b/feature_level_1 --feature_level 1 --skip_train

# python render.py -s /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/data/09c1414f1b -m output/09c1414f1b/feature_level_2 --feature_level 2 --skip_train

python render.py -s /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/data/09c1414f1b -m output_test_split/09c1414f1b/feature_level_3 --feature_level 3 --skip_train --eval

