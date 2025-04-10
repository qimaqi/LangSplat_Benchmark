#!/bin/bash
#SBATCH --job-name=data_process_scannetpp_30_40
#SBATCH --output=sbatch_log/data_process_scannetpp_30_40_%j.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1

#SBATCH --nodelist=bmicgpu06,bmicgpu07,bmicgpu08,bmicgpu09,bmicgpu10,octopus01,octopus02,octopus03,octopus04
#SBATCH --cpus-per-task=4
#SBATCH --mem 128GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qi.ma@vision.ee.ethz.ch
source /scratch_net/schusch/qimaqi/miniconda3/etc/profile.d/conda.sh
conda activate langsplat_cu18_test

cd /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/LangSplat_Qi/

# python preprocess.py --dataset_path /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/LangSplat_Qi/data/lerf_ovs/ramen

python preprocess_scannetpp.py --dataset_path /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/data --start_idx 30 --end_idx 40




# --dataset_path /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/data
    # parser.add_argument('--dataset_path', type=str, required=True)
    # parser.add_argument('--resolution', type=int, default=-1)
    # parser.add_argument('--sam_ckpt_path', type=str, default="ckpts/sam_vit_h_4b8939.pth")