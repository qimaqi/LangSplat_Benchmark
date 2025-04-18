#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import concurrent.futures
# from colorama import Fore, init, Style
from tqdm import tqdm
import glob
from scene.colmap_read_write import read_images_text

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]

        image = Image.open(image_path)
     
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    except:
        colors = np.random.rand(positions.shape[0], positions.shape[1])
    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        normals = np.random.rand(positions.shape[0], positions.shape[1])
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

    # extract mesh information
    # mesh.compute_vertex_normals(normalized=True)
    # coord = np.array(mesh.vertices).astype(np.float32)
    # color = (np.array(mesh.vertex_colors) * 255).astype(np.uint8)
    # normal = np.array(mesh.vertex_normals).astype(np.float32)

    # return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    # reading_dir_F = "language_feature" if language_feature == None else language_feature
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        print("##### Colmap dataset evaluation mode ######")
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        print("#### Using all cameras for training ####")
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            # cam_name_F = os.path.join(path, frame["file_path"] + "") # TODO: extension?
            
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            # language_feature_path = os.path.join(path, cam_name_F)
            # language_feature_name = Path(cam_name_F).stem
            # language_feature = Image.open(language_feature_path) # TODO: data read

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,  
                              image_path=image_path, 
                              image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


# def readCamerasFromTransforms_DL3DV(path, transformsfile, extension=".png"):
#     cam_infos = []
#     with open(os.path.join(path, transformsfile)) as json_file:
#         contents = json.load(json_file)
#         # try:
#         #     fovx = contents["camera_angle_x"]
#         # except:
#         #     fovx = None
#         fx = contents["fl_x"] / 4
#         fy = contents["fl_y"] / 4
#         cx = contents["cx"] / 4
#         cy = contents["cy"] / 4
#         w = contents["w"] / 4
#         h = contents["h"] / 4
#         fovx = 2 * np.arctan(w / (2 * fx))


#         # "w": 3840,
#         # "h": 2160,
#         # "fl_x": 1678.070706876354,
#         # "fl_y": 1685.4932539564572,
#         # "cx": 1920.0,
#         # "cy": 1080.0,
#         # "k1": -0.002225639890808761,
#         # "k2": 0.00021814239516741203,
#         # "p1": 0.00039974709199719095,
#         # "p2": 0.0001251200581788733,
#         frames = contents["frames"]
#         # check if filename already contain postfix
#         if frames[0]["file_path"].split('.')[-1] in ['jpg', 'jpeg', 'JPG', 'png']:
#             extension = ""

#         def process_frame(idx, frame):
#             # Process each frame and append cam_info to cam_infos list
#             cam_name = frame["file_path"] + extension
#             cam_name = cam_name.replace("images/", "images_4/")
#             image_path = os.path.join(path, cam_name)
#             if not os.path.exists(image_path):
#                 raise ValueError(f"Image {image_path} does not exist!")
#             # NeRF 'transform_matrix' is a camera-to-world transform
#             c2w = np.array(frame["transform_matrix"])

#             # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
#             c2w[:3, 1:3] *= -1

#             # get the world-to-camera transform and set R, T
#             w2c = np.linalg.inv(c2w)

#             R = np.transpose(
#                 w2c[:3, :3]
#             )  # R is stored transposed due to 'glm' in CUDA code
#             T = w2c[:3, 3]

#             image_name = Path(cam_name).stem
#             image = Image.open(image_path)

#             # if (
#             #     "k1" in frame
#             #     and "k2" in frame
#             #     and "p1" in frame
#             #     and "p2" in frame
#             #     and "k3" in frame
#             # ):
#             #     mtx = np.array(
#             #         [
#             #             [frame["fl_x"], 0, frame["cx"]],
#             #             [0, frame["fl_y"], frame["cy"]],
#             #             [0, 0, 1.0],
#             #         ],
#             #         dtype=np.float32,
#             #     )
#             #     dist = np.array(
#             #         [frame["k1"], frame["k2"], frame["p1"], frame["p2"], frame["k3"]],
#             #         dtype=np.float32,
#             #     )
#             #     im_data = np.array(image.convert("RGB"))
#             #     arr = cv2.undistort(im_data / 255.0, mtx, dist, None, mtx)
#             #     image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

#             if fovx is not None:
#                 fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
#                 FovY = fovy
#                 FovX = fovx
#             else:
#                 # given focal in pixel unit
#                 FovY = focal2fov(frame["fl_y"], image.size[1])
#                 FovX = focal2fov(frame["fl_x"], image.size[0])

#             return CameraInfo(
#                 uid=idx,
#                 R=R,
#                 T=T,
#                 FovY=FovY,
#                 FovX=FovX,
#                 image=image,
#                 image_path=image_path,
#                 image_name=image_name,
#                 width=image.size[0],
#                 height=image.size[1],
#             )
#         ct = 0
#         progress_bar = tqdm(frames, desc="Loading dataset")

#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             # 提交每个帧到执行器进行处理
#             futures = [executor.submit(process_frame, idx, frame) for idx, frame in enumerate(frames)]

#             # 使用as_completed()获取已完成的任务
#             for future in concurrent.futures.as_completed(futures):
#                 cam_info = future.result()
#                 cam_infos.append(cam_info)
                
#                 ct+=1
#                 if ct % 10 == 0:
#                     progress_bar.set_postfix({"num": Fore.YELLOW+f"{ct}/{len(frames)}"+Style.RESET_ALL})
#                     progress_bar.update(10)

#             progress_bar.close()
    
#     cam_infos = sorted(cam_infos, key = lambda x : x.image_name)
#     return cam_infos

# def readDL3DVSyntheticInfo(path, eval, extension=".png"):
#     print("Reading Training Transforms")
#     total_cam_infos = readCamerasFromTransforms_DL3DV(path, "transforms.json", extension)
#     print("Reading Test Transforms")
#     test_indices = np.arange(0, len(total_cam_infos), len(total_cam_infos) // 10)
#     train_indices = set(np.arange(0, len(total_cam_infos))) - set(test_indices)

#     train_cam_infos = [total_cam_infos[i] for i in train_indices]
#     test_cam_infos = [total_cam_infos[i] for i in test_indices]

#     if not eval:
#         train_cam_infos.extend(test_cam_infos)
#         test_cam_infos = []

#     nerf_normalization = getNerfppNorm(train_cam_infos)

#     scene_info = SceneInfo(point_cloud=pcd,
#                            train_cameras=train_cam_infos,
#                            test_cameras=test_cam_infos,
#                            nerf_normalization=nerf_normalization,
#                            ply_path=ply_path)
#     return scene_info


# def readCamerasFromTransforms_SCANNETPP(path, transformsfile, extension=".png"):
#     cam_infos = []
#     meta_path = path / "dslr/nerfstudio/transforms_undistorted.json"
#     data_root = path
#     with open(meta_path, "r") as fp:
#         meta = json.load(fp)
    
#         resize = tuple(meta.get("resize", [584, 876]))
        
#         new_h = resize[0]
#         new_w = resize[1]
#         resize_ratio = resize[1] / meta.get("w", 1752)

#         fx = meta["fl_x"] * resize_ratio
#         fy = meta["fl_y"] * resize_ratio
#         cx = meta["cx"] * resize_ratio
#         cy = meta["cy"] * resize_ratio

#         fovx = 2 * np.arctan(new_w / (2 * fx))
        

#         frames = meta["frames"] + meta.get("test_frames", [])
#         colmap_dir = path / "dslr/colmap"
#         images_txt_path = colmap_dir / "images.txt"


#         images = read_images_text(images_txt_path)
#         images_name_2_id = {image.name: image.id for image in images.values()}

#         # check if filename already contain postfix
#         # if frames[0]["file_path"].split('.')[-1] in ['jpg', 'jpeg', 'JPG', 'png']:
#         #     extension = ""

#         def process_frame(idx, frame, downsampling=False):
#             # Process each frame and append cam_info to cam_infos list
#             frame_name = frame["file_path"]
#             image_id = images_name_2_id[frame_name]

#             pose = np.eye(4)
#             rot = images[image_id].qvec2rotmat()
#             pose[:3, :3] = rot
#             pose[:3, 3] = images[image_id].tvec
#             c2w = np.linalg.inv(pose) 
            

#             # image_path = os.path.join(path, cam_name)
#             # if not os.path.exists(image_path):
#             #     raise ValueError(f"Image {image_path} does not exist!")
#             # # NeRF 'transform_matrix' is a camera-to-world transform
#             # c2w = np.array(frame["transform_matrix"])

#             # # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
#             # c2w[:3, 1:3] *= -1

#             # get the world-to-camera transform and set R, T
#             w2c = np.linalg.inv(c2w)

#             R = np.transpose(
#                 w2c[:3, :3]
#             )  # R is stored transposed due to 'glm' in CUDA code
#             T = w2c[:3, 3]

#             # image_name = Path(cam_name).stem
#             image_name = frame_name
#             image_path = data_root / ("dslr/undistorted_images/" + frame_name)
#             image = Image.open(image_path)
#             resize_img = (
#                 (resize[1], resize[0])
#                 if resize[1] > resize[0]
#                 else (resize[0], resize[1])
#             )
#             image = image.resize(resize_img, Image.Resampling.LANCZOS)

#             image = np.array(image.convert("RGB"))  # Ensure RGB format
#             image = Image.fromarray(image)

#             # depth_path = data_root / ("dslr/undistorted_depths/" + frame_name)
#             # depth_path = str(depth_path).replace("JPG", "png")

#             # depth = Image.open(depth_path)
#             # depth_tensor = torch.from_numpy(np.array(depth).astype(np.float32))
#             # depth_tensor = depth_tensor / 1000.0
#             # depth_tensor = torch.nn.functional.interpolate(
#             #     depth_tensor.unsqueeze(0).unsqueeze(0), size=resize, mode="nearest"
#             # ).squeeze()    

#             if fovx is not None:
#                 fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
#                 FovY = fovy
#                 FovX = fovx
#             else:
#                 # given focal in pixel unit
#                 FovY = focal2fov(frame["fl_y"], image.size[1])
#                 FovX = focal2fov(frame["fl_x"], image.size[0])

#             return CameraInfo(
#                 uid=idx,
#                 R=R,
#                 T=T,
#                 FovY=FovY,
#                 FovX=FovX,
#                 image=image,
#                 # depth=depth_tensor,
#                 image_path=image_path,
#                 image_name=image_name,
#                 width=image.size[0],
#                 height=image.size[1],
#             )
#         ct = 0
#         progress_bar = tqdm(frames, desc="Loading dataset")

#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             # 提交每个帧到执行器进行处理
#             futures = [executor.submit(process_frame, idx, frame) for idx, frame in enumerate(frames)]

#             # 使用as_completed()获取已完成的任务
#             for future in concurrent.futures.as_completed(futures):
#                 cam_info = future.result()
#                 cam_infos.append(cam_info)
                
#                 ct+=1
#                 if ct % 10 == 0:
#                     progress_bar.set_postfix({"num": Fore.YELLOW+f"{ct}/{len(frames)}"+Style.RESET_ALL})
#                     progress_bar.update(10)

#             progress_bar.close()
    
#     cam_infos = sorted(cam_infos, key = lambda x : x.image_name)
#     return cam_infos

# def readSCANNETPPSyntheticInfo(path, eval, extension=".png", warmup_ply_path=None):
#     print("Reading Training Transforms")
#     path = Path(path)

#     total_cam_infos = readCamerasFromTransforms_SCANNETPP(path, "transforms.json", extension)
#     print("Reading Test Transforms")
#     test_indices = np.arange(0, len(total_cam_infos), len(total_cam_infos) // 10)
#     train_indices = set(np.arange(0, len(total_cam_infos))) - set(test_indices)

#     train_cam_infos = [total_cam_infos[i] for i in train_indices]
#     test_cam_infos = [total_cam_infos[i] for i in test_indices]

#     if not eval:
#         train_cam_infos.extend(test_cam_infos)
#         test_cam_infos = []

#     nerf_normalization = getNerfppNorm(train_cam_infos)
 
#     ply_paths = glob.glob(os.path.join(path, "scans", "*.ply"))
#     assert len(ply_paths) > 0,f"Could not find any ply files in {os.path.join(path, 'scans')}"
#     ply_path = ply_paths[0]
#     pcd = fetchPly(ply_path)

#     scene_info = SceneInfo(point_cloud=pcd,
#                            train_cameras=train_cam_infos,
#                            test_cameras=test_cam_infos,
#                            nerf_normalization=nerf_normalization,
#                            ply_path=ply_path)
#     return scene_info

def readCamerasFromTransforms_nerfstudio(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(transformsfile) as json_file:
        contents = json.load(json_file)
        focal_len_x = contents["fl_x"]
        focal_len_y = contents["fl_y"]
        cx = contents["cx"] * 2 
        cy = contents["cy"] * 2 
        fovx = focal2fov(focal_len_x, cx)
        fovy = focal2fov(focal_len_y, cy)

        FovY = fovy 
        FovX = fovx
        frames = contents["frames"]
        # raise ValueError("Frames: ", frames)
        for idx, frame in enumerate(frames):
            image_path = path.replace("nerfstudio", "undistorted_images")
            cam_name = frame["file_path"]

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            applied_transform = np.array([
                [0,  1,  0,  0],
                [1,  0,  0,  0],
                [0,  0, -1,  0],
                [0,  0,  0,  1],
            ], dtype=float)
            c2w = np.dot(applied_transform, c2w)
            # get the world-to-camera transform and set R, T
            # w2c = c2w
            w2c = np.linalg.inv(c2w)
            w2c[1:3] *= -1
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            image_path = os.path.join(image_path, cam_name)
            image_name = Path(cam_name).stem
    

            image = Image.open(image_path)
            # resize
            resize = [584, 876]
            resize_img = (
                (resize[1], resize[0])
                if resize[1] > resize[0]
                else (resize[0], resize[1])
            )
            # image = image.resize(resize_img, Image.Resampling.LANCZOS)
            image = image.resize(resize_img, Image.LANCZOS)
            
            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""
            # , depth_path=depth_path
            # cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
            #                 image_path=image_path, image_name=image_name,
            #                 width=cx, height=cy, depth_path=depth_path, depth_params=None, is_test=is_test))
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image=image, image_path=image_path, image_name=image_name,
                            width=cx, height=cy))
            
    return cam_infos

def readScanNetppInfo(path, white_background, depths, eval, lang_path, llff_hold=8, extension=".JPG", ):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    all_cam_infos = readCamerasFromTransforms_nerfstudio(path, lang_path, depths_folder, white_background, False, extension)
    # print("Reading Test Transforms")
    # test_cam_infos = readCamerasFromTransforms_nerfstudio(path, lang_path, depths_folder, white_background, True, extension)
    
    if not eval:
        train_cam_infos = all_cam_infos
        test_cam_infos = []
    else:
        train_cam_infos = [c for idx, c in enumerate(all_cam_infos) if idx % llff_hold != 0]
        test_cam_infos = [c for idx, c in enumerate(all_cam_infos) if idx % llff_hold == 0]

    # train_cam_infos = [c for idx, c in enumerate(all_cam_infos) if idx % llff_hold != 0]
    # test_cam_infos = [c for idx, c in enumerate(all_cam_infos) if idx % llff_hold == 0]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                        #    is_nerf_synthetic=False
                           )
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    # "DL3DV": readDL3DVSyntheticInfo,
    # "SCANNETPP": readSCANNETPPSyntheticInfo
    "ScanNetpp": readScanNetppInfo
}