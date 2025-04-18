#!/bin/bash
CASE_NAME="09c1414f1b"

# path to lerf_ovs/label
gt_folder="/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/data/09c1414f1b/dslr/segmentation_2d"

root_path="/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/LangSplat_Qi"

source /scratch_net/schusch/qimaqi/miniconda3/etc/profile.d/conda.sh
conda activate langsplat_cu18_test

python evaluate_iou_loc_npy.py \
        --dataset_name ${CASE_NAME} \
        --feat_dir ${root_path}/output_test_split \
        --ae_ckpt_dir ${root_path}/autoencoder/ckpt \
        --output_dir ${root_path}/eval_result_test_split \
        --mask_thresh 0.4 \
        --encoder_dims 256 128 64 32 3 \
        --decoder_dims 16 32 64 128 256 256 512 \
        --gt_npy_folder ${gt_folder} \
        --label_name_txt /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/metadata/semantic_benchmark/top100.txt

# /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/data/09c1414f1b