import os
import glob

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import hsv_to_rgb
from openclip_encoder import OpenCLIPNetwork
import sys
sys.path.append("..")
from autoencoder.model import Autoencoder

dataset_root = '/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/'
device = "cuda"
ae_ckpt_path = '/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/LangSplat_Qi/autoencoder/ckpt/09c1414f1b/best_ckpt.pth'
# instantiate autoencoder and openclip
clip_model = OpenCLIPNetwork(device)
checkpoint = torch.load(ae_ckpt_path, map_location=device)
ae_model = Autoencoder([256, 128, 64, 32, 3], [16, 32, 64, 128, 256, 256, 512]).to(device)
ae_model.load_state_dict(checkpoint)
ae_model.eval()




segment_class_names = np.loadtxt(
    Path(dataset_root) / "metadata" / "semantic_benchmark" / "top100.txt",
    dtype=str,
    delimiter=".",  # dummy delimiter to replace " "
)
def generate_distinct_colors(n=100, seed=42):
    np.random.seed(seed)
    
    # Evenly space hues, randomize saturation and value a bit
    hues = np.linspace(0, 1, n, endpoint=False)
    np.random.shuffle(hues)  # shuffle to prevent similar colors being close in order
    saturations = np.random.uniform(0.6, 0.9, n)
    values = np.random.uniform(0.7, 0.95, n)
    
    hsv_colors = np.stack([hues, saturations, values], axis=1)
    rgb_colors = hsv_to_rgb(hsv_colors)
    return rgb_colors

# Example usage
SCANNET_100_COLORS = generate_distinct_colors(100)
CLASS_LABELS_100 =  segment_class_names


# from metadata.scannet200_constants import CLASS_LABELS_20

# SCANNET_20_COLORS = np.array([
#     [174, 199, 232],  # wall
#     [152, 223, 138],  # floor
#     [31, 119, 180],   # cabinet
#     [255, 187, 120],  # bed
#     [188, 189, 34],   # chair
#     [140, 86, 75],    # sofa
#     [255, 152, 150],  # table
#     [214, 39, 40],    # door
#     [197, 176, 213],  # window
#     [148, 103, 189],  # bookshelf
#     [196, 156, 148],  # picture
#     [23, 190, 207],   # counter
#     [247, 182, 210],  # desk
#     [219, 219, 141],  # curtain
#     [255, 127, 14],   # refrigerator
#     [227, 119, 194],  # shower curtain
#     [158, 218, 229],  # toilet
#     [44, 160, 44],    # sink
#     [112, 128, 144],  # bathtub
#     [82, 84, 163],    # other furniture
# ], dtype=np.uint8)  # shape: (20, 3)
# # Convert to [0,1] float range for matplotlib
# SCANNET_20_COLORS = SCANNET_20_COLORS / 255.0

def main():
    scene_folder = "/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/LangSplat_Qi/output_test_split/09c1414f1b/feature_level_1/test/ours_None/gt_npy"
    gt_2d_folder = "/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/data/09c1414f1b/dslr/segmentation_2d"
    text_emb_path = "/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/metadata/semantic_benchmark/clip_text_embeddings_100.pth"
    text_embeds = torch.load(text_emb_path)  # shape (20, 768)
    assert text_embeds.shape[-1] == 512
    text_embeds_np = text_embeds.detach().cpu().numpy()  # shape (100, 768)

    out_folder = os.path.join(scene_folder, "gt_render_zero_shot_semseg_level_1")
    os.makedirs(out_folder, exist_ok=True)

    seg_files = sorted(glob.glob(os.path.join(scene_folder, "*.npy")))
    for seg_path in seg_files:
        # Derive the matching feature path
        base_name = os.path.basename(seg_path).replace(".npy", "")
        # feat_path = os.path.join(scene_folder, base_name + "_f.npy")
        seg_file_name = os.path.basename(seg_path)
        feat_path = seg_path
        gt_label_path = os.path.join(gt_2d_folder, seg_file_name)

        if not os.path.exists(feat_path):
            print(f"Feature file not found for {seg_path}, skipping.")
            continue

        # Load segmentation and features
        feat = np.load(feat_path) # shape (N_segments, 768)
        feat = torch.from_numpy(feat).to(device)
        feat = ae_model.decode(feat)
        print("feat shape: ", feat.shape)
        feat = feat.detach().cpu().numpy()  # shape (N_segments, 768)
        print("gt_label_path", gt_label_path)
        gt_label = np.load(gt_label_path)   # shape (1, H, W)
        # print("gt_label shape: ", gt_label.shape)
        # print("feat shape: ", feat.shape)
        feat = feat.reshape(-1, 512)

        logits = np.dot(text_embeds_np, feat.T)  # shape: (20, N_segments)
        scores = 1 / (1 + np.exp(-logits))  # shape: (20, N_segments)
        pred_labels = np.argmax(scores, axis=0)  # shape: (N_segments,)


        H, W = gt_label.shape
        pred_labels = pred_labels.reshape(H, W)

        print("pred_labels shape: ", pred_labels.shape, pred_labels.min(), pred_labels.max())
        pred_class_map = pred_labels

        # pred_class_map = -1 * np.ones((H, W), dtype=np.int32)  # will store 0..19 for valid classes, -1 otherwise

        # unique_seg_ids = np.unique(gt_label)
        # for s_id in unique_seg_ids:
        #     if s_id < 0:
        #         # -1 indicates background/ignored
        #         continue
        #     if s_id >= len(pred_labels):
        #         # Safety check if there's any mismatch
        #         print(f"Warning: segment ID {s_id} out of range for features.")
        #         continue
        #     s_id = int(s_id)  # Ensure s_id is an integer
        #     # print(f"Processing segment ID: {s_id}")
        #     class_idx = pred_labels[s_id]
        #     pred_class_map[gt_label == s_id] = class_idx

        # raise Exception("Debugging")

        # ----------------------------------------------
        # Convert predicted class map => color image
        # ----------------------------------------------
        # pred_class_map is shape (H, W), values in [0..19] or -1
        pred_color = np.zeros((H, W, 3), dtype=np.float32)
        valid_mask = (pred_class_map >= 0)

        # Index into palette for valid pixels
        pred_color[valid_mask] = SCANNET_100_COLORS[pred_class_map[valid_mask]]

        # If you want the ignored pixels to appear black or something else:
        # pred_color[~valid_mask] = [0.0, 0.0, 0.0]  # black

        # ----------------------------------------------
        # Visualization with overlaid text
        # ----------------------------------------------
        plt.figure(figsize=(10, 8))
        plt.imshow(pred_color)
        plt.axis("off")


        out_path = os.path.join(out_folder, base_name + ".jpg")
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close()



        gt_color = np.zeros((H, W, 3), dtype=np.float32)
        valid_mask = (gt_label >= 0)

        # Index into palette for valid pixels
        gt_color[valid_mask] = SCANNET_100_COLORS[gt_label[valid_mask]]


        plt.figure(figsize=(10, 8))
        plt.imshow(gt_color)
        plt.axis("off")


        # unique_segs = np.unique(gt_label)
        # for s_id in unique_segs:
        #     if s_id < 0:
        #         # -1 indicates ignored areas
        #         continue
        #     # Gather the mask for this segment
        #     mask = (gt_label == s_id)
        #     coords = np.argwhere(mask)
        #     if coords.size == 0:
        #         continue

        #     # Centroid of the segment in (y, x)
        #     cy, cx = coords.mean(axis=0)

        #     # print(pred_labels)
        #     # print(s_id)
        #     s_id = int(s_id)
        #     class_idx = pred_labels[s_id]
        #     class_name = CLASS_LABELS_100[class_idx]
        #     plt.text(
        #         x=cx, y=cy,
        #         s=class_name,
        #         color="white",
        #         fontsize=8,
        #         ha="center",
        #         va="center",
        #         bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.1")
        #     )

        out_path = os.path.join(out_folder, base_name + "_gt.jpg")
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close()

        print(f"Saved zero-shot seg visualization to: {out_path}")

if __name__ == "__main__":
    main()