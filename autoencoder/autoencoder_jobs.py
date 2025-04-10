import os 
import sys 
import argparse
import numpy as np
from tqdm import tqdm
import json
import shutil

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


    for val_name_i in tqdm(val_split):
        print("Processing", val_name_i)
        source_path = os.path.join(root_path, val_name_i, 'dslr')
        selected_json = os.path.join(source_path, 'nerfstudio', 'lang_feat_selected_imgs.json')
        with open(selected_json, 'r') as f:
            selected_data_list = json.load(f)
        selected_frames = selected_data_list['frames']
        selected_imgs_list = [frame_i['file_path'] for frame_i in selected_frames]
        selected_imgs_name_list = [img_i.split('.JPG')[0] for img_i in selected_imgs_list]
        # output_path = os.path.join('/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/LangSplat_Qi/output', val_name_i)
        # os.makedirs(output_path, exist_ok=True)
        # output_path_feature_level = os.path.join(output_path, 'feature_level')
        # potential_output_file = os.path.join(output_path, 'feature_level_3/chkpnt30000.pth')
        # if os.path.exists(potential_output_file):
        #     print("potential_output_file exists", potential_output_file)
        #     continue
        # check if language feature available
        input_language_features = os.path.join(source_path, 'language_features')
        if not os.path.exists(input_language_features):
            print("input_language_features not exists", input_language_features)
            continue
        else:
            # assert all files is created in preprocess part
            finished_imgs_list = os.listdir(input_language_features)
            

            finished_imgs_list = os.listdir(input_language_features)
            finished_names_list = [img_i.split('_')[0] for img_i in finished_imgs_list]

            finished_names_list = np.unique(finished_names_list)
            # print("finished_names_list", len(finished_names_list), len(selected_imgs_name_list))

            finished_names_list = sorted(finished_names_list)
            selected_imgs_name_list = sorted(selected_imgs_name_list)
            # check if the finished_names_list and selected_imgs_name_list are equal
            if finished_names_list != selected_imgs_name_list:
                print("misatch", finished_names_list, selected_imgs_name_list)
                continue
        
            for npy_file_i in os.listdir(input_language_features):
                if npy_file_i.endswith('.npy'):
                    npy_file_i_path = os.path.join(input_language_features, npy_file_i)
                    # check if the file is in the selected_imgs_name_list
                    file_name = os.path.basename(npy_file_i_path).split('.')[0]
                    file_name = file_name.split('_')[0]
                    # if file_name not in selected_imgs_name_list:
                    #     print("file_name not in selected_imgs_name_list", file_name)
                    #     # shutil remove 
                        # print("remove file", npy_file_i_path)
                        # os.remove(npy_file_i_path)
             

        output_langauge_features = os.path.join(source_path, 'language_features_dim3')
        # if os.path.exists(output_langauge_features) and len(os.listdir(output_langauge_features)) > 0:
        #     print("output_langauge_features exists", output_langauge_features)
        #     continue

        # now run the command
        if not os.path.exists(os.path.join('/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/LangSplat_Qi/autoencoder/ckpt', val_name_i, 'best_ckpt.pth')):
            command = "python train.py --dataset_path {} --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name {}".format(source_path, val_name_i)
            print("command", command)
            # os.system(command) 
        else:
            print("Trianing finish already", val_name_i)

        
        if not (os.path.exists(output_langauge_features) and len(os.listdir(output_langauge_features)) > 0) :
            test_command = "python test.py --dataset_path {} --dataset_name {}".format(source_path, val_name_i)
            print("test_command", test_command)
            # os.system(test_command)
        else:
            print("Testing finish already", val_name_i)

# python test.py --dataset_path /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/data/27dd4da69e/dslr  --dataset_name 27dd4da69e 


# python train.py --dataset_path /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/data/27dd4da69e/dslr --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007  --dataset_name 27dd4da69e

