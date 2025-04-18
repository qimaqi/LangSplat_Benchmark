import os 
import shutil
import numpy as np
split_path = '/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/splits/nvs_sem_val.txt'

val_split = np.loadtxt(split_path, dtype=str)
# print("val_split", val_split)
val_split = sorted(val_split)

val_split_10_scene = []
val_split_10_scene.extend(val_split[:2])
val_split_10_scene.extend(val_split[10:12])
val_split_10_scene.extend(val_split[20:22])
val_split_10_scene.extend(val_split[30:32])
val_split_10_scene.extend(val_split[40:42])

print("val_split_10_scene", val_split_10_scene)

tgt_path = '/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/subset_sequences/'

for val_name_i in val_split_10_scene:
    shutil.copytree(os.path.join('/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/data', val_name_i), os.path.join(tgt_path, val_name_i)) 
    print("copying", val_name_i, 'to', tgt_path)
