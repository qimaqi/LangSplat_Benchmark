import os 
import sys 
import argparse
import numpy as np



def get_config():
    args= argparse.ArgumentParser(description="Train a model")
    args.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="start idx to start with",
    )
    args.add_argument(
        "--end_idx",
        type=int,
        default=-1,
        help="end idx to end with",
    )
    args.add_argument(
        "--root_path",
        type=str,
        default='/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/data/',
        help="end idx to end with",
    )
    args.add_argument(
        "--split_path",
        type=str,
        default='/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/splits/nvs_sem_val.txt'
    )

    # finsih argsparser
    args = args.parse_args()
    return args



if __name__ == "__main__":
    cfgs = get_config()
    start_idx = cfgs.start_idx
    end_idx = cfgs.end_idx
    split_path = cfgs.split_path
    root_path = cfgs.root_path
    val_split = np.loadtxt(split_path, dtype=str)
    # print("val_split", val_split)
    val_split = sorted(val_split)

    if end_idx == -1:
        end_idx = len(val_split)

    val_split = val_split[start_idx:end_idx]

    for val_name_i in val_split:
        source_path = os.path.join(root_path, val_name_i)
        output_path = os.path.join('/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/LangSplat_Qi/output', val_name_i)
        os.makedirs(output_path, exist_ok=True)
        output_path_feature_level = os.path.join(output_path, 'feature_level')
        potential_output_file = os.path.join(output_path, 'feature_level_3/chkpnt30000.pth')
        if os.path.exists(potential_output_file):
            print("potential_output_file exists", potential_output_file)
            continue

        ply_path = os.path.join('/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannetpp_v1_fix_gs/', val_name_i, 'ckpts', 'point_cloud_30000.ply')
        if not os.path.exists(ply_path):
            print("ply_path not exists", ply_path)
            continue
        
        for feature_level in range(1, 3):
            cmd = f"python train_ply.py -s {source_path} -m {output_path_feature_level} --start_ply {ply_path} --feature_level {feature_level}"
            print("cmd", cmd)
            os.system(cmd)
            # break