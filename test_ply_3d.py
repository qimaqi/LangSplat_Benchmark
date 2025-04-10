"""
Tester

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import time
import numpy as np
from collections import OrderedDict
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data
import copy
from scipy.spatial import cKDTree

import argparse
# from pointcept.utils.misc import (
#     AverageMeter,
#     intersection_and_union,
#     intersection_and_union_gpu,
#     make_dirs,
#     neighbor_voting,
#     clustering_voting
# )
import json
from dataclasses import dataclass, field
from typing import Tuple, Type

import torchvision
import open_clip
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Module

import open3d as o3d
import pandas as pd
from pathlib import Path
from autoencoder.model import Autoencoder

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

def intersection_and_union(output, target, K, ignore_index=-1):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def _majority_vote(neighbor_labels, ignore_label, num_classes):
    """
    neighbor_labels: (N, K) array, each row are the K neighbor labels for a point
    ignore_label: int, label to treat as "invalid" that we ignore in counting
    num_classes: total number of valid classes (>= ignore_label+1 if ignore is negative, 
                or just the max label + 1)
    Returns:
    out: (N,) array of majority‐voted labels
    """
# We define the numba‐accelerated function inside so it can capture arguments
    @numba.njit(parallel=True)
    def _vote(labels_2d, out):
        n_points = labels_2d.shape[0]
        k = labels_2d.shape[1]
        for i in numba.prange(n_points):
            counts = np.zeros(num_classes, dtype=np.int32)
            max_count = 0
            best_label = ignore_label
            # Count only valid labels
            for j in range(k):
                lbl = labels_2d[i, j]
                if lbl != ignore_label and 0 <= lbl < num_classes:
                    counts[lbl] += 1
            # Get majority
            for c in range(num_classes):
                if counts[c] > max_count:
                    max_count = counts[c]
                    best_label = c
            out[i] = best_label

    n_points = neighbor_labels.shape[0]
    out = np.full((n_points,), ignore_label, dtype=np.int32)
    _vote(neighbor_labels, out)
    return out


def neighbor_voting(coords, pred, vote_k, ignore_label, num_classes, valid_mask=None):
    """
    coords:       (N, 3) array of all points
    pred:         (N,)   array of predicted labels for each point
    vote_k:       int, number of neighbors to fetch for each point
    ignore_label: int, label for 'invalid' or 'ignored'
    num_classes:  int, total # of "real" classes (not counting ignore_label)
    valid_mask:   (N,) bool array, optional. 
                  If provided, we build the KD‐tree only on coords[valid_mask],
                  but we still query for neighbors for *all* N points.

    Returns:
      new_pred: (N,) array of updated predictions after neighbor voting.
                If the majority of neighbors are ignore_label, the result is ignore_label.
    """
    if valid_mask is not None:
        used_coords = coords[valid_mask]
        used_labels = pred[valid_mask]
        print(f"Using valid_mask {len(used_coords)}/{len(coords)} points for voting")
    else:
        used_coords = coords
        used_labels = pred

    if len(used_coords) == 0:
        return pred

    kd_tree = cKDTree(used_coords)
    # Query neighbors for ALL points (including those that fail valid_mask or are ignore_label)
    # nn_indices will be shape (N, vote_k)
    _, nn_indices = kd_tree.query(coords, k=vote_k)
    neighbor_labels = used_labels[nn_indices]

    new_pred = _majority_vote(
            neighbor_labels, ignore_label, num_classes
        )
    return new_pred



@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)

class OpenCLIPNetwork(nn.Module):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,  # e.g., ViT-B-16
            pretrained=self.config.clip_model_pretrained,  # e.g., laion2b_s34b_b88k
            precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

        self.positives = self.config.positives    
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        assert (
            self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
            self.pos_embeds.shape[1] == self.clip_n_dims
        ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims
    
    def gui_cb(self,element):
        self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[:, 0, :]

    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)


    def encode_scannetpp_text(self, text_path, save_path):
        with open(text_path, "r") as f:
            self.class_names = [line.strip() for line in f if line.strip()]

        tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.class_names]).to("cuda")
        text_embedding = self.model.encode_text(tok_phrases)
        text_embedding = text_embedding.float()
        # save text embedding
        text_embedding = text_embedding.cpu()
        print("text_embedding", text_embedding.shape)
        text_embedding = F.normalize(text_embedding, p=2, dim=1)
        torch.save(text_embedding, save_path)
    


class ZeroShotLangSplat3DTester(object):
    def __init__(self, 
                 enable_voting=False,
                 vote_k=25,
                 confidence_threshold=0.1,
                 class_names="/insait/qimaqi/data/scannetpp_v1_fixed/metadata/semantic_benchmark/top100.txt",
                 text_embeddings="/insait/qimaqi/data/scannetpp_v1_fixed/metadata/semantic_benchmark/clip_text_embeddings.pth",
                 ):
        if 'index' in kwargs:
            # multi-dataset testing
            cfg = copy.deepcopy(cfg)
            cfg['test'] = cfg['test'][kwargs['index']]
            cfg.data.test = cfg.data.test[kwargs['index']]
        self.cfg = cfg
        self.enable_voting = cfg['test'].get('enable_voting', enable_voting)
        self.vote_k = cfg['test'].get('vote_k', vote_k)
        self.confidence_threshold = cfg['test'].get('confidence_threshold', confidence_threshold)
        self.save_feat = cfg['test'].get('save_feat', save_feat)
        self.skip_eval = cfg['test'].get('skip_eval', skip_eval)
        self.ignore_index = ignore_index

        class_names = cfg['test'].get('class_names', class_names)
        text_embeddings = cfg['test'].get('text_embeddings', text_embeddings)
        excluded_classes = cfg['test'].get('excluded_classes', excluded_classes)
        
        # Load class names and text embeddings
        with open(class_names, "r") as f:
            self.class_names = [line.strip() for line in f if line.strip()]
        self.text_embeddings = torch.load(text_embeddings).cuda()
        self.text_embeddings = F.normalize(self.text_embeddings, p=2, dim=1)
        
        # Handle excluded classes
        self.excluded_indices = [i for i, name in enumerate(self.class_names) 
                               if name in (excluded_classes or [])]
        self.keep_indices = [i for i in range(len(self.class_names)) 
                            if i not in self.excluded_indices]
        self.num_keep_classes = len(self.keep_indices)
        self.num_classes = len(self.class_names)
        assert self.num_classes == self.text_embeddings.size(0), "Mismatch in class names and text embeddings"

    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>> ZeroShotSemSegTester Start Evaluation >>>>>>>>>>>>>")
        logger.info(f"Testing on {self.cfg.data.test.split} split of {self.cfg.data.test.type}")
        logger.info(f"ZeroShotSemSegTester Loaded text embeddings with shape {self.text_embeddings.shape}")
        if self.enable_voting:
            logger.info("Neighbor voting enabled with k={}".format(self.vote_k))
        if self.save_feat:
            logger.info("Saving inference feature enabled")
        if self.skip_eval:
            logger.info("Skipping evaluation")
        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, f"result_{self.cfg.data.test.type}")
        make_dirs(save_path)
        
        # ================ Preserved Submission Handling ================
        # Create submit folder only on main process
        if (
            self.cfg.data.test.type == "ScanNetDataset"
            or self.cfg.data.test.type == "ScanNet200Dataset"
            or self.cfg.data.test.type == "ScanNetPPDataset"
            or 'ScanNetPP' in self.cfg.data.test.type
            or 'ScanNet' in self.cfg.data.test.type
        ) and comm.is_main_process():
            make_dirs(os.path.join(save_path, "submit"))
        elif (
            self.cfg.data.test.type == "SemanticKITTIDataset" and comm.is_main_process()
        ):
            make_dirs(os.path.join(save_path, "submit"))
        elif self.cfg.data.test.type == "NuScenesDataset" and comm.is_main_process():
            import json
            make_dirs(os.path.join(save_path, "submit", "lidarseg", "test"))
            make_dirs(os.path.join(save_path, "submit", "test"))
            submission = dict(
                meta=dict(
                    use_camera=False,
                    use_lidar=True,
                    use_radar=False,
                    use_map=False,
                    use_external=False,
                )
            )
            with open(
                os.path.join(save_path, "submit", "test", "submission.json"), "w"
            ) as f:
                json.dump(submission, f, indent=4)
        if self.save_feat:
            make_dirs(os.path.join(save_path, "feat"))
        comm.synchronize()
        # ================ End Submission Handling ================
        record = {}
        # Fragment inference
        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]  # assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")
            pred_save_path = os.path.join(save_path, f"{data_name}_pred.npy")
            feat_save_path = os.path.join(save_path, "feat", f"{data_name}_feat.pth") if self.save_feat else None

            if os.path.isfile(pred_save_path) and not self.save_feat:
                logger.info(f"{data_name}: loaded existing pred")
                pred = np.load(pred_save_path)
                if "origin_segment" in data_dict:
                    segment = data_dict["origin_segment"]
                if self.cfg.data.test.type == "ScanNetPPDataset" or 'ScanNetPP' in self.cfg.data.test.type:
                    pred = pred[:, 0] # we save top-3 classes for ScanNetPP
            else:
                num_points = segment.size
                num_classes = self.text_embeddings.size(0)
                ignore_index = self.ignore_index

                # Create a buffer to accumulate probabilities (or logits)
                pred = torch.zeros((num_points, num_classes), device="cuda")
                pred_coord = torch.zeros((num_points, 3), device="cuda")

                if self.save_feat:
                    feat_dim = self.text_embeddings.shape[1]  # Get feature dimension
                    point_features = torch.zeros((num_points, feat_dim), device="cuda")
                    feature_counts = torch.zeros(num_points, device="cuda")

                # ---------------------------------------------------------------------
                # Accumulate probabilities from each fragment
                # ---------------------------------------------------------------------
                for i, frag_dict in enumerate(fragment_list):
                    # collate => partial data
                    input_dict = collate_fn([frag_dict])
                    # move to GPU
                    for key in input_dict:
                        if isinstance(input_dict[key], torch.Tensor):
                            input_dict[key] = input_dict[key].cuda(non_blocking=True)

                    idx_part = input_dict["index"]
                    offset_list = input_dict["offset"]

                    # Forward pass
                    with torch.no_grad():
                        out_dict = self.model(input_dict, chunk_size=600000)
                        # e.g., point feature [M, feat_dim]
                        pred_part_feat = out_dict["point_feat"]["feat"]  # shape [M, feat_dim]
                        logits = torch.mm(pred_part_feat, self.text_embeddings.t())
                        pred_part_prob = torch.sigmoid(logits)  # [M, num_classes]

                    # Accumulate into the large buffer
                    bs = 0
                    for be in offset_list:
                        # sum up probabilities
                        pred[idx_part[bs:be], :] += pred_part_prob[bs:be]
                        # track coords if needed
                        pred_coord[idx_part[bs:be]] = input_dict["coord"][bs:be]
                        if self.save_feat:
                            point_features[idx_part[bs:be], :] += pred_part_feat[bs:be]
                            feature_counts[idx_part[bs:be]] += 1

                        bs = be
                    
                    logger.info(
                        f"Test: {idx+1}/{len(self.test_loader)}-{data_name}, "
                        f"Fragment batch: {i}/{len(fragment_list)}"
                    )
                
                if self.save_feat:
                    pred_mask = feature_counts > 0
                    # points appear multiple times together in fragments
                    point_features[pred_mask] /= feature_counts[pred_mask].unsqueeze(1)
                    if "origin_segment" in data_dict and "inverse" in data_dict:
                        point_features_cpu = point_features.cpu()
                        point_features_cpu = F.normalize(point_features_cpu, p=2, dim=1)
                        final_features = point_features_cpu[data_dict["inverse"]]
                    else:
                        final_features = F.normalize(point_features, p=2, dim=1)
                    torch.save(final_features, feat_save_path)
                    logger.info(f"Saved pred feature with shape {final_features.shape} to {feat_save_path}")
                    del point_features, final_features 
                
                if "ScanNetPP" in self.cfg.data.test.type:
                    # e.g. we want top-3 classes for each point
                    pred = pred.topk(3, dim=1)[1].cpu().numpy()  # shape => [N, 3]
                else:
                    # typical semantic seg => pick best
                    max_probs, argmax_indices = torch.max(pred, dim=1)
                    argmax_indices[max_probs < self.confidence_threshold] = ignore_index
                    pred = argmax_indices.cpu().numpy()
                
                if "origin_segment" in data_dict:
                    assert "inverse" in data_dict, "Inverse mapping is required to map pred to full origin_coord"
                    pred = pred[data_dict["inverse"]]  # shape => [original_num_points, ...]
                    segment = data_dict["origin_segment"]

                np.save(pred_save_path, pred)
                # ================  Submission Saving ================
                if (
                    self.cfg.data.test.type in ["ScanNetDataset", "ScanNet200Dataset", "ScanNetGSDataset", "ScanNet200GSDataset"]
                ):
                    np.savetxt(
                        os.path.join(save_path, "submit", f"{data_name}.txt"),
                        self.test_loader.dataset.class2id[pred].reshape([-1, 1]),
                        fmt="%d",
                    )
                elif self.cfg.data.test.type == "ScanNetPPDataset" or 'ScanNetPP' in self.cfg.data.test.type:
                    np.savetxt(
                        os.path.join(save_path, "submit", f"{data_name}.txt"),
                        pred.astype(np.int32),
                        delimiter=",",
                        fmt="%d",
                    )
                    pred = pred[:, 0] if pred.ndim > 1 else pred  # Handle 1D/2D pred
                elif self.cfg.data.test.type == "SemanticKITTIDataset":
                    sequence_name, frame_name = data_name.split("_")
                    os.makedirs(
                        os.path.join(save_path, "submit", "sequences", 
                                   sequence_name, "predictions"),
                        exist_ok=True,
                    )
                    submit = pred.astype(np.uint32)
                    submit = np.vectorize(
                        self.test_loader.dataset.learning_map_inv.__getitem__
                    )(submit).astype(np.uint32)
                    submit.tofile(
                        os.path.join(
                            save_path, "submit", "sequences",
                            sequence_name, "predictions", f"{frame_name}.label"
                        )
                    )
                elif self.cfg.data.test.type == "NuScenesDataset":
                    np.array(pred + 1).astype(np.uint8).tofile(
                        os.path.join(
                            save_path, "submit", "lidarseg", "test",
                            f"{data_name}_lidarseg.bin"
                        )
                    )
            if self.skip_eval:    
                continue
            # ---------------------------------------------------------------------
            # Apply neighbor voting if enabled
            if self.enable_voting:
                num_classes = self.num_classes
                ignore_index = self.ignore_index
                if "origin_coord" in data_dict:
                    coords = data_dict["origin_coord"]
                    pred = neighbor_voting(coords, pred, self.vote_k, ignore_index, num_classes, valid_mask=data_dict.get("origin_feat_mask", None))
                else:
                    logger.warning("Neighbor voting requires 'origin_coord' in data_dict, skipped..")
                if "origin_instance" in data_dict:
                    pred = clustering_voting(pred, data_dict["origin_instance"], ignore_index)

            intersection, union, target = intersection_and_union(
                pred, segment, self.num_classes, self.ignore_index
            )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)

            record[data_name] = dict(
                intersection=intersection,
                union=union,
                target=target
            )

            # Per‐scene IoU & accuracy
            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target) + 1e-10)

            # Running average across scenes so far
            mask_union = union_meter.sum != 0
            mask_target = target_meter.sum != 0
            m_iou = np.mean((intersection_meter.sum / (union_meter.sum + 1e-10))[mask_union])
            m_acc = np.mean((intersection_meter.sum / (target_meter.sum + 1e-10))[mask_target])

            batch_time.update(time.time() - end)
            logger.info(
                f"Test: {data_name} [{idx+1}/{len(self.test_loader)}]-{segment.size} "
                f"Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                f"Accuracy {acc:.4f} ({m_acc:.4f}) "
                f"mIoU {iou:.4f} ({m_iou:.4f})"
            )

        if self.skip_eval: 
            logger.info("<<<<<<<<<<<<<<<<< Tester End, Skipped Evaluation <<<<<<<<<<<<<<<<<")
            return

        # ---------------------------------------------------------------------
        # Sync across processes if distributed
        # ---------------------------------------------------------------------
        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            # Merge results
            final_record = {}
            while record_sync:
                r = record_sync.pop()
                final_record.update(r)
                del r    
            self._log_final_metrics(final_record, save_path)

    @staticmethod
    def collate_fn(batch):
        return batch

    def _log_final_metrics(self, final_record, save_path):
        logger = get_root_logger()
        
        intersection = np.sum([v["intersection"] for v in final_record.values()], axis=0)
        union = np.sum([v["union"] for v in final_record.values()], axis=0)
        target = np.sum([v["target"] for v in final_record.values()], axis=0)

        iou_class = intersection / (union + 1e-10)
        accuracy_class = intersection / (target + 1e-10)

        mask_union = union != 0
        mask_target = target != 0
        mIoU = np.mean(iou_class[mask_union])
        mAcc = np.mean(accuracy_class[mask_target])
        allAcc = sum(intersection) / (sum(target) + 1e-10)

        logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                mIoU, mAcc, allAcc
            )
        )

        # foreground metrics (excluding classes in excluded_classes)
        if self.excluded_indices:
            fg_iou_class = iou_class[self.keep_indices]
            fg_accuracy_class = accuracy_class[self.keep_indices]
            fg_mask_union = union[self.keep_indices] != 0
            fg_mask_target = target[self.keep_indices] != 0
            fg_mIoU = np.mean(fg_iou_class[fg_mask_union])
            fg_mAcc = np.mean(fg_accuracy_class[fg_mask_target])

            # foreground allAcc
            fg_intersection = intersection[self.keep_indices]
            fg_target = target[self.keep_indices]
            fg_allAcc = sum(fg_intersection) / (sum(fg_target) + 1e-10)

            logger.info(
                "Foreground Val result (excluding {} classes): mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                    len(self.excluded_indices), fg_mIoU, fg_mAcc, fg_allAcc
                )
            )

        # Optionally log per‐class results
        if self.class_names:
            for i, cls_name in enumerate(self.class_names):
                logger.info(
                    f"Class_{i}-{cls_name} Result: iou/accuracy "
                    f"{iou_class[i]:.4f}/{accuracy_class[i]:.4f}"
                )
        else:
            for i in range(self.num_classes):
                logger.info(
                    f"Class_{i} iou/accuracy "
                    f"{iou_class[i]:.4f}/{accuracy_class[i]:.4f}"
                )

        # Save results to a text file
        result_file = os.path.join(save_path, "eval_results.txt")
        with open(result_file, "w") as f:
            f.write(
                "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}\n".format(
                    mIoU, mAcc, allAcc
                )
            )

            # Write foreground metrics
            if self.excluded_indices:
                f.write(
                    "Foreground Val result (excluding {} classes): mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}\n".format(
                        len(self.excluded_indices), fg_mIoU, fg_mAcc, fg_allAcc
                    )
                )

            # Write per-class results
            f.write("\nPer-class results:\n")
            if self.class_names:
                for i, cls_name in enumerate(self.class_names):
                    f.write(
                        "Class_{}-{} Result: iou/accuracy {:.4f}/{:.4f}\n".format(
                            i, cls_name, iou_class[i], accuracy_class[i]
                        )
                    )
            else:
                for i in range(self.num_classes):
                    f.write(
                        "Class_{} iou/accuracy {:.4f}/{:.4f}\n".format(
                            i, iou_class[i], accuracy_class[i]
                        )
                    )

            # Mark which classes were excluded
            if self.excluded_indices:
                f.write("\nExcluded classes:\n")
                for idx in self.excluded_indices:
                    if hasattr(self.cfg.data, "names"):
                        f.write(f"Class_{idx}-{self.cfg.data.names[idx]}\n")
                    else:
                        f.write(f"Class_{idx}\n")

        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        
def filter_map_classes(mapping, count_thresh, count_type, mapping_type):
    if count_thresh > 0 and count_type in mapping.columns:
        mapping = mapping[mapping[count_type] >= count_thresh]
    if mapping_type == "semantic":
        map_key = "semantic_map_to"
    elif mapping_type == "instance":
        map_key = "instance_map_to"
    else:
        raise NotImplementedError
    # create a dict with classes to be mapped
    # classes that don't have mapping are entered as x->x
    # otherwise x->y
    map_dict = OrderedDict()

    for i in range(mapping.shape[0]):
        row = mapping.iloc[i]
        class_name = row["class"]
        map_target = row[map_key]

        # map to None or some other label -> don't add this class to the label list
        try:
            if len(map_target) > 0:
                # map to None -> don't use this class
                if map_target == "None":
                    pass
                else:
                    # map to something else -> use this class
                    map_dict[class_name] = map_target
        except TypeError:
            # nan values -> no mapping, keep label as is
            if class_name not in map_dict:
                map_dict[class_name] = class_name

    return map_dict


def get_argparse():
    parser = argparse.ArgumentParser(description="Test a model")
    parser.add_argument(
        "--gs_path", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--gt_path_root", type=str, required=True, help="Path to the ground truth file"
    )
    parser.add_argument(
        "--dataset_root", type=str, default='/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full', help="Path to the dataset root"
    )

    args = parser.parse_args()
    return args


# python test_ply_3d.py --gs_path /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/LangSplat_Qi/output/27dd4da69e_1/chkpnt30000.pth --gt_path_root /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/data/27dd4da69e/scans


if __name__ == '__main__':
    args = get_argparse()
    model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
    text_embed_save_path = '/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/metadata/semantic_benchmark/clip_text_embeddings_100.pth'
    text_path = '/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/metadata/semantic_benchmark/top100.txt'
    dataset_root = args.dataset_root
    confidence_threshold = 0.1
    vote_k = 25

    segment_class_names = np.loadtxt(
        Path(args.dataset_root) / "metadata" / "semantic_benchmark" / "top100.txt",
        dtype=str,
        delimiter=".",  # dummy delimiter to replace " "
    )

    if not os.path.exists(text_embed_save_path):
        print("Encoding text embeddings...")
        model.encode_scannetpp_text(text_path, text_embed_save_path)
    else:
        print("Text embeddings already encoded, loading...")

    text_embeddings = torch.load(text_embed_save_path).cuda()
    # run 3d tester

    # Question, for gaussian language feature, is there -1 ?
    # we load the gaussian xyz and gaussian langauge feature in 512 dimension

    model_params, first_iter = torch.load(args.gs_path) 
    (active_sh_degree, 
    _xyz, 
    _features_dc, 
    _features_rest,
    _scaling, 
    _rotation, 
    _opacity,
    _language_feature,
    max_radii2D, 
    xyz_gradient_accum, 
    denom,
    opt_dict, 
    spatial_lr_scale) = model_params
    # load gt

    gt_mesh_path = os.path.join(args.gt_path_root, "mesh_aligned_0.05.ply")
    gt_segs_path = os.path.join(args.gt_path_root, "segments.json")
    gt_anno_path = os.path.join(args.gt_path_root, "segments_anno.json")

    mesh = o3d.io.read_triangle_mesh(str(gt_mesh_path))
    coord = np.array(mesh.vertices).astype(np.float32)

    with open(gt_segs_path) as f:
        segments = json.load(f)

    with open(gt_anno_path) as f:
        anno = json.load(f)
    seg_indices = np.array(segments["segIndices"], dtype=np.uint32)
    num_vertices = len(seg_indices)
    assert num_vertices == len(coord)
    ignore_index = -1
    semantic_gt = np.ones((num_vertices, 3), dtype=np.int16) * ignore_index
    instance_gt = np.ones((num_vertices, 3), dtype=np.int16) * ignore_index

    label_mapping = pd.read_csv(
        Path(args.dataset_root) / "metadata" / "semantic_benchmark" / "map_benchmark.csv"
    )
    label_mapping = filter_map_classes(
        label_mapping, count_thresh=0, count_type="count", mapping_type="semantic"
    )
    class2idx = {
        class_name: idx for (idx, class_name) in enumerate(segment_class_names)
    }

    # number of labels are used per vertex. initially 0
    # increment each time a new label is added
    instance_size = np.ones((num_vertices, 3), dtype=np.int16) * np.inf
    labels_used = np.zeros(num_vertices, dtype=np.int16)

    for idx, instance in enumerate(anno["segGroups"]):
        label = instance["label"]
        instance["label_orig"] = label
        # remap label
        instance["label"] = label_mapping.get(label, None)
        instance["label_index"] = class2idx.get(label, ignore_index)

        if instance["label_index"] == ignore_index:
            continue
        # get all the vertices with segment index in this instance
        # and max number of labels not yet applied
        mask = np.isin(seg_indices, instance["segments"]) & (labels_used < 3)
        size = mask.sum()
        if size == 0:
            continue

        # get the position to add the label - 0, 1, 2
        label_position = labels_used[mask]
        semantic_gt[mask, label_position] = instance["label_index"]
        # store all valid instance (include ignored instance)
        instance_gt[mask, label_position] = instance["objectId"]
        instance_size[mask, label_position] = size
        labels_used[mask] += 1

    # major label is the label of smallest instance for each vertex
    # use major label for single class segmentation
    # shift major label to the first column
    mask = labels_used > 1
    if mask.sum() > 0:
        major_label_position = np.argmin(instance_size[mask], axis=1)

        major_semantic_label = semantic_gt[mask, major_label_position]
        semantic_gt[mask, major_label_position] = semantic_gt[:, 0][mask]
        semantic_gt[:, 0][mask] = major_semantic_label

        major_instance_label = instance_gt[mask, major_label_position]
        instance_gt[mask, major_label_position] = instance_gt[:, 0][mask]
        instance_gt[:, 0][mask] = major_instance_label

    # semantic_gt shape N
    # instance_gt shape N
    assert len(semantic_gt) == len(_language_feature), f"semantic_gt: {len(semantic_gt)}, _language_feature: {len(_language_feature)}"

    semantic_gt = semantic_gt[:, 0]
    
    data_name = os.path.basename(os.path.dirname(args.gs_path))
    save_path = os.path.join(args.gt_path_root, data_name)

    ae_checkpoint = os.path.join('/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/LangSplat_Qi/autoencoder/ckpt', data_name.split("_")[0], 'best_ckpt.pth')

    assert os.path.exists(ae_checkpoint), f"Autoencoder checkpoint not found at {ae_checkpoint}"

    os.makedirs(save_path, exist_ok=True)
    pred_save_path = os.path.join(save_path, f"{data_name}_pred.npy")
    feat_save_path = os.path.join(save_path, "feat", f"{data_name}_feat.pth") 
    
    # Create a buffer to accumulate probabilities (or logits)
    num_points = semantic_gt.size
    num_classes = text_embeddings.size(0)

    pred = torch.zeros((num_points, num_classes), device="cuda")
    pred_coord = torch.zeros((num_points, 3), device="cuda")

    semantic_gt = torch.from_numpy(semantic_gt).cuda()
    print("_language_feature", _language_feature.shape)

    pred_part_feat = _language_feature.cuda()
    # find pred_part_feat with 0,0,0
    pred_part_feat_sum = torch.sum(torch.abs(pred_part_feat), dim=-1)
    pred_part_feat_mask = pred_part_feat_sum > 0
    print("Nonzero pred_part_feat", pred_part_feat_mask.sum(), 'of', pred_part_feat.shape[0])
    pred_part_feat_mask = pred_part_feat_mask.cpu().numpy()
    # autoencder back to 512 dimension
    encoder_hidden_dims = [256, 128, 64, 32, 3]
    decoder_hidden_dims = [16, 32, 64, 128, 256, 256, 512]

    ae_model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")
    ae_checkpoint_load = torch.load(ae_checkpoint)
    ae_model.load_state_dict(ae_checkpoint_load)
    ae_model.eval()

    with torch.no_grad():
        # lvl, h, w, _ = sem_feat.shape
        restored_feat = ae_model.decode(pred_part_feat)
        # restored_feat = restored_feat.view(lvl, h, w, -1)           # 3x832x1264x512

    # pred_part_feat_decode = ae_model.encode(pred_part_feat).to("cpu").numpy()  
    print("restored_feat", restored_feat.shape)
    # renormalize
    restored_feat = F.normalize(restored_feat, p=2, dim=-1)
    # pred_part_feat = out_dict["point_feat"]["feat"]  # shape [M, feat_dim]
    # max_probs, argmax_indices = torch.max(logits, dim=1)
    # argmax_indices[max_probs < confidence_threshold] = ignore_index
    # pred = argmax_indices.cpu().numpy()

    restored_feat_mask = restored_feat[pred_part_feat_mask]
    gt_mask = semantic_gt[pred_part_feat_mask]
    logits = torch.mm(restored_feat_mask, text_embeddings.t()).softmax(dim=-1)

    # pred = logits.topk(3, dim=1)[1].cpu().numpy()  # shape => [N, 3]
    pred = logits.topk(1, dim=1)[1].cpu().numpy()  # shape => [N, 3]

    coord = coord[pred_part_feat_mask]
    pred = neighbor_voting(coord, pred, vote_k, ignore_index, num_classes, valid_mask=None)

    print("after voting pred", pred.shape)
    print("gt_mask", gt_mask.shape)
    gt_mask = gt_mask.cpu().numpy()

    intersection, union, target = intersection_and_union(
        pred, gt_mask, num_classes, ignore_index
    )
    record = {}
    record[data_name] = dict(
        intersection=intersection,
        union=union,
        target=target
    )

    # Per‐scene IoU & accuracy
    mask = union != 0
    iou_class = intersection / (union + 1e-10)
    iou = np.mean(iou_class[mask])
    acc = sum(intersection) / (sum(target) + 1e-10)

    print("iou_class", iou_class)
    print("acc", acc)
    print("iou", iou)

    # acc 0.10824689671320656
    # iou 0.012283376296118304

    # # Running average across scenes so far
    # mask_union = union_meter.sum != 0
    # mask_target = target_meter.sum != 0
    # m_iou = np.mean((intersection_meter.sum / (union_meter.sum + 1e-10))[mask_union])
    # m_acc = np.mean((intersection_meter.sum / (target_meter.sum + 1e-10))[mask_target])
    
    # if "origin_coord" in data_dict:
    #     coords = data_dict["origin_coord"]
        
    # else:
    #     logger.warning("Neighbor voting requires 'origin_coord' in data_dict, skipped..")
    # if "origin_instance" in data_dict:
    #     pred = clustering_voting(pred, data_dict["origin_instance"], ignore_index)

# intersection, union, target = intersection_and_union(
#     pred, segment, self.num_classes, self.ignore_index
# )
# intersection_meter.update(intersection)
# union_meter.update(union)
# target_meter.update(target)