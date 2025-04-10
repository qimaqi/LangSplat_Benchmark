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
import numpy as np
import torch

import os
from tqdm import tqdm
# from os import makedirs
# from gaussian_renderer import render
# import torchvision
# from utils.general_utils import safe_state
# from argparse import ArgumentParser
# from arguments import ModelParams, PipelineParams, get_combined_args
# from gaussian_renderer import GaussianModel
import open3d as o3d
import argparse
import json 
import math 
import pandas as pd
from pathlib import Path
from PIL import Image
from collections import OrderedDict
import cv2
import kaolin as kal
from scipy.ndimage import distance_transform_edt

import torch
import pytorch3d
try:
    from pytorch3d.renderer.mesh import rasterize_meshes
    from pytorch3d.renderer.mesh.rasterizer import RasterizationSettings
    from pytorch3d.renderer.mesh.shader import SoftDepthShader # Used for perspective correct Z
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import TexturesUV, MeshRasterizer, MeshRenderer, HardPhongShader, PointLights, RasterizationSettings, BlendParams
    from pytorch3d.renderer.mesh.shading import interpolate_face_attributes
    from pytorch3d.renderer import (
        look_at_view_transform,
        FoVPerspectiveCameras,
        PerspectiveCameras,
        SoftPhongShader,
    )
    _has_pytorch3d = True
except ImportError:
    _has_pytorch3d = False
    print("WARNING: PyTorch3D not found. Optimized rasterization is unavailable.")


from pytorch3d.structures import Meshes


def render_depth_plus_pytorch3d(face_vertices_image, face_vertices_z, semantics, camera_params , device, render_scale=1.0):
        """
        Rasterize a depth map, face index map, and UV map using PyTorch3D for speed.

        Args:
        - face_vertices_image: Tensor (1, num_faces, 3, 2) with 2D vertices (NDC XY) of each face. Assumes +Y is up.
        - face_vertices_z: Tensor (1, num_faces, 3) with view-space z-values (positive depth) of each face vertex.
        - view_target_h: Height of the target image
        - view_target_w: Width of the target image
        - device: Device to use
        - uv_faces: Optional Tensor (1, num_faces, 3, 2) with uv coordinates of each face vertex.
        - render_scale: Scale factor for rendering resolution (default 1.0).

        Returns:
        - depth_image: Tensor (1, view_target_h, view_target_w, 1)
        - face_idx_buffer: Tensor (1, view_target_h, view_target_w)
        - uv_image: Tensor (1, view_target_h, view_target_w, 2)
        """
        batch_size = 1 # This function processes one mesh at a time
        num_faces = face_vertices_image.shape[1]
        render_h = int(camera_params['h'] * render_scale)
        render_w = int(camera_params['w'] * render_scale)

        face_vertices_image = face_vertices_image.to(device)
        face_vertices_z = face_vertices_z.to(device)

        # PyTorch3D's NDC conventions (+X right, +Y up, +Z into screen)        
        face_vertices_image[..., 0] *= -1       # Flip X to match PyTorch3D's conventions
        verts_packed = torch.cat(
            (face_vertices_image, face_vertices_z.unsqueeze(-1)),
            dim=-1
        ) # Shape: (1, num_faces, 3, 3)
        # print("verts_packed", verts_packed.shape) # verts_packed torch.Size([1, 6905073, 3, 3])

        verts_list = [verts_packed[0].reshape(-1, 3)]
        faces_list = [torch.arange(num_faces * 3, device=device).reshape(num_faces, 3)]
        pytorch3d_mesh = Meshes(verts=verts_list, faces=faces_list)

        # Adjust bin_size and max_faces_per_bin for performance/memory trade-off, `bin_size=0` coarse-to-fine approach is often fast.
        raster_settings = RasterizationSettings(
            image_size=(render_h, render_w), blur_radius=0.0, faces_per_pixel=1, perspective_correct=True, cull_backfaces=False, clip_barycentric_coords=False)

        # - pix_to_face: (N, H, W, faces_per_pixel) LongTensor mapping pixels to face indices (-1 for background)
        # - zbuf: (N, H, W, faces_per_pixel) FloatTensor depth buffer (lower values are closer)
        # - bary_coords: (N, H, W, faces_per_pixel, 3) FloatTensor barycentric coordinates
        pix_to_face, zbuf, bary_coords, _ = rasterize_meshes(
            meshes=pytorch3d_mesh,
            image_size=(render_h, render_w),
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            perspective_correct=raster_settings.perspective_correct,
            clip_barycentric_coords=raster_settings.clip_barycentric_coords,
            cull_backfaces=raster_settings.cull_backfaces
        )

        face_idx_buffer = pix_to_face.squeeze(-1)
        # face_idx_buffer = torch.where(face_idx_buffer >= 0, face_idx_buffer, torch.tensor(0, dtype=torch.long, device=device))

        depth_image = zbuf.squeeze(-1).unsqueeze(-1)
        background_mask = (pix_to_face.squeeze(-1) < 0).unsqueeze(-1) # if no face in this pixel, set depth to 0
        depth_image = torch.where(background_mask, torch.tensor(0.0, device=device), depth_image)
        depth_image = torch.clamp(depth_image, min=0.0)

        semantic_image = torch.zeros((1, render_h, render_w), dtype=torch.int32, device=device) - 1  # -1 for background
        # Fill in the semantic image with face indices
        # face_vertex_labels = semantics[faces_list[0]] # num_faces x 3
        # print("face_vertex_labels", face_vertex_labels, face_vertex_labels.min(), face_vertex_labels.max())
        face_vertex_labels_voting = torch.zeros((num_faces, 1), dtype=torch.int32, device=device) # num_faces x 1
        for face_idx in range(num_faces):
            vertices_i = semantics[faces_list[0][face_idx]] # 3 x 1
            print("vertices_i", vertices_i, vertices_i.min(), vertices_i.max())

        raise NotImplementedError 
            # unique_labels, unique_counts = torch.unique(face_vertex_labels[face_idx], return_counts=True)
            # max_count_index = torch.argmax(unique_counts)
            # face_vertex_labels_voting[face_idx] = unique_labels[max_count_index] # num_faces x 1

        # unique_labels, unique_counts = torch.unique(face_vertex_labels, return_counts=True, dim=1)
        # max_count_index = torch.argmax(unique_counts, dim=1)
        # face_vertex_labels_voting = unique_labels[torch.arange(num_faces, device=device), max_count_index] # num_faces x 1

#             # max voting, if 3 vertices are different, using the first one
#             unique_labels, unique_counts = np.unique(vertices_semantics_label, return_counts=True)
#             max_count_index = np.argmax(unique_counts)
#             semantic_map[h_idx, w_idx] = unique_labels[max_count_index]

        for h_idx in range(render_h):
            for w_idx in range(render_w):
                if face_idx_buffer[0, h_idx, w_idx] >= 0:
                    face_idx = face_idx_buffer[0, h_idx, w_idx]
                    semantic_image[0, h_idx, w_idx] = face_vertex_labels_voting[face_idx]

        semantic_image = semantic_image.squeeze(0) # HxW
     
        # if render_scale != 1.0:        # Handle potential upsampling if render_scale was not 1.0
        #     depth_image = F.interpolate(depth_image.permute(0, 3, 1, 2), size=(view_target_h, view_target_w), mode='nearest').permute(0, 2, 3, 1)
        #     face_idx_buffer = F.interpolate(face_idx_buffer.unsqueeze(1).float(), size=(view_target_h, view_target_w), mode='nearest').long().squeeze(1)
        #     uv_image = F.interpolate(uv_image.permute(0, 3, 1, 2), size=(view_target_h, view_target_w), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

        z_buf_to_save = depth_image.cpu().numpy()
        z_buf_to_save_normalize = z_buf_to_save.copy()
        z_buf_to_save_normalize = (z_buf_to_save - z_buf_to_save.min()) / (z_buf_to_save.max() - z_buf_to_save.min())
        z_buf_to_save_normalize = (255 * z_buf_to_save_normalize).astype(np.uint8)
        z_buf_to_save_normalize = cv2.applyColorMap(z_buf_to_save_normalize, cv2.COLORMAP_JET)
        z_buf_to_save_normalize = Image.fromarray(z_buf_to_save_normalize)
        z_buf_to_save_normalize.save('./z_buf.png')

        raise ValueError("debug depth", depth_image.shape, face_idx_buffer.shape, semantic_image.shape)


        return depth_image, semantic_image



def prepare_vertices_ndc(vertices, faces, camera_transform, camera_params, device='cuda'):
    """
    Transforms vertices to clip space and then NDC, another method than projection with intrinsic matrix, used to scale with custom 3*focal_x

    Args:
        vertices: 3D vertices in world space.
        faces: Face indices.
        projection_matrix: The camera's projection matrix (4x4).
        camera_transform: The camera's transformation matrix (3x4).

    Returns:
        face_vertices_camera: Vertices in camera space.
        face_vertices_ndc: Vertices in NDC space.
        face_normals: Face normals.
    """
    vertices = vertices.to(device)
    faces = faces.to(device).to(torch.int64)
    camera_transform = camera_transform.to(device)

    img_size= [camera_params['w'], camera_params['h']]
    padded_vertices = torch.nn.functional.pad(vertices, (0, 1), mode='constant', value=1.) # Nx4
    if len(camera_transform.shape) == 2:
        camera_transform = camera_transform.unsqueeze(0)
    if camera_transform.shape[1] == 4:        # want 3x4
        camera_transform = camera_transform[:, :3, :].transpose(1, 2) # 
    vertices_camera = (padded_vertices @ camera_transform) # Nx4 @ 4x3 = Nx3

    # Apply projection matrix (camera_transform to clip space)
    near = 0.1
    far = 100.
    cx = img_size[0] / 2 
    cy = img_size[1] / 2 
    fx = camera_params['focal_len_x']
    fy = camera_params['focal_len_y']
    fovx = camera_params['fovx']
    fovy = camera_params['fovy']

    # NOTE: through tests I found that 3*focal_x is the correct values, why?
    # projection_matrix = torch.tensor([[3*intrinsics.focal_x/self.img_size[0], 0, 2*cx/self.img_size[0]-1, 0],
    #                                     [0, 2*intrinsics.focal_y/self.img_size[1], 2*cy/self.img_size[1]-1, 0],
    #                                     [0, 0, -(far+near)/(far-near), -2*far*near/(far-near)],
    #                                     [0, 0, -1, 0]], device=vertices.device)

    tanhalffov_x = math.tan((fovx/2)) # radians
    tanhalffov_y = math.tan((fovy/2)) # radians
    s1 = 1/tanhalffov_x
    s2 = 1/tanhalffov_y

    # To map z to the range [0, 1] use:
    f1 =  far / (far - near)
    f2 = -(far * near) / (far - near)

    # Projection matrix
    projection_matrix = [
            [s1,   0,   0,   0],
            [0,   s2,   0,   0],
            [0,    0,   f1,  f2],
            [0,    0,    1,   0],
    ]
    print("projection_matrix", projection_matrix)

    projection_matrix = torch.tensor(projection_matrix, device=vertices.device).float().to(device) # 4x4
    
    vertices_camera_pad = torch.nn.functional.pad(vertices_camera, (0, 1), mode='constant', value=1.)
    vertices_camera_pad = vertices_camera_pad.unsqueeze(0)
    vertices_clip = torch.matmul(vertices_camera_pad, projection_matrix.transpose(0, 1))
    
    # Perform W-division (clip space to NDC)
    vertices_clip = vertices_clip.squeeze(0)        # (B, N, 4)
    # print("vertices_clip", vertices_clip.shape)
    vertices_ndc = vertices_clip[:, :, :3] / (vertices_clip[:, :, 3].unsqueeze(2) + 1e-6)
    
    # Get face vertices in camera and NDC space
    face_vertices_camera = kal.ops.mesh.index_vertices_by_faces(vertices_camera, faces)
    face_vertices_ndc = kal.ops.mesh.index_vertices_by_faces(vertices_ndc, faces)
    face_normals = kal.ops.mesh.face_normals(face_vertices_camera, unit=True)

    # print("face_vertices_camera", face_vertices_camera.shape)
    # print("face_vertices_ndc", face_vertices_ndc.shape)
    # face_vertices_camera torch.Size([1, 6905073, 3, 3])
    # face_vertices_ndc torch.Size([1, 6905073, 3, 3])
    # print("face_vertices_camera", face_vertices_camera.min(), face_vertices_camera.max())
    # print("face_vertices_ndc", face_vertices_ndc[:, :, :, :2].min(), face_vertices_ndc[:, :, :, :2].max())

    # :storage-server: Useful commands:Add user_A read & write access to /data/folder Step1: setfacl -R -m u:user_A:rwx /data/folder/Step2: setfacl -R -m mask::rwx /data/folder/Step3: setfacl -R -d -m u:user_A:rwx /data/folder/\

    # raise ValueError("debug face_vertices_camera", face_vertices_camera.shape, face_vertices_ndc.shape)
    return face_vertices_camera, face_vertices_ndc[:, :, :, :2], face_normals



def opencv_to_pytorch3d(c2w):
    # pytorch 3d is x point to left, y point to up, z point to front
    # opencv is x point to right, y point to down, z point to front
    cv2p3d = np.array([[-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
    c2w = c2w @ cv2p3d

    return c2w


def project_mesh_to_2d(verts, faces, semantics,image_size, camera_params, device='cuda'):
    """
    Project mesh to 2D and get vertex-to-pixel mapping
    
    Args:
        verts: (V, 3) tensor of mesh vertices
        faces: (F, 3) tensor of mesh faces
        semantics: (V,) tensor of vertex semantic labels
        image_size: (height, width) tuple
        camera_params: dict containing:
            - 'R': (3, 3) rotation matrix
            - 'T': (3, 1) translation vector
            - 'focal_length': scalar or (fx, fy)
            - 'principal_point': (px, py)
        device: torch device
    
    Returns:
        - rendered_image: (H, W) tensor with face indices (-1 for background)
        - vertex_pixel_coords: (V, 2) tensor of vertex pixel coordinates
        - vertex_visibility: (V,) bool tensor indicating visible vertices
    """
    # Create mesh structure
    mesh = Meshes(verts=[verts], faces=[faces]).to(device)
    
    # Setup camera
    h, w = image_size
    R = camera_params['R'].unsqueeze(0).to(device)  # (1, 3, 3)
    T = camera_params['T'].unsqueeze(0).to(device)  # (1, 3)
    focal_length = camera_params['focal_length'].unsqueeze(0).to(device)  # (1, 2)
    principal_point = camera_params['principal_point'].unsqueeze(0).to(device)  # (1, 2)
    fov_x = camera_params.get('fovx', 0.0)
    fov_y = camera_params.get('fovy', 0.0)
    # intrinsic_matrix = camera_params.get('K', None).unsqueeze(0).to(device)  # (1, 4, 4)
    
    # PyTorch3D's NDC conventions (+X right, +Y up, +Z into screen)        
    face_vertices_image[..., 0] *= -1       # Flip X to match PyTorch3D's conventions
    verts_packed = torch.cat(
        (face_vertices_image, face_vertices_z.unsqueeze(-1)),
        dim=-1
    ) # Shape: (1, num_faces, 3, 3)

    verts_list = [verts_packed[0].reshape(-1, 3)]
    faces_list = [torch.arange(num_faces * 3, device=device).reshape(num_faces, 3)]

    pytorch3d_mesh = Meshes(verts=verts_list, faces=faces_list)


    pix_to_face, zbuf, bary_coords, _ = rasterize_meshes(
        meshes=pytorch3d_mesh,
        image_size=(render_h, render_w),
        blur_radius=raster_settings.blur_radius,
        faces_per_pixel=raster_settings.faces_per_pixel,
        perspective_correct=raster_settings.perspective_correct,
        clip_barycentric_coords=raster_settings.clip_barycentric_coords,
        cull_backfaces=raster_settings.cull_backfaces
    )

    # cameras = FoVPerspectiveCameras(
    #     R=R, T=T,
    #     znear=0.1,
    #     zfar=100.0,
    #     aspect_ratio=1.0,
    #     fov=fov_y, # fov_y
    #     degrees=False,
    #     # K = intrinsic_matrix,
    # )
    
    # cameras = PerspectiveCameras(
    #     focal_length=torch.tensor([focal_length[0], focal_length[1]]).reshape(1, 2),
    #     principal_point=torch.tensor([principal_point[0], principal_point[1]]).reshape(1, 2),
    #     R=R,
    #     T=T,
    #     device=device,
    #     image_size=torch.tensor([h, w]).reshape(1, 2),

    # )

    # # Rasterization settings - we only need face indices
    # raster_settings = RasterizationSettings(
    #     image_size=image_size,
    #     blur_radius=0.0,
    #     faces_per_pixel=1,  # Only closest face per pixel
    #     perspective_correct=False,
    # )
    
    # # Create rasterizer
    # rasterizer = MeshRasterizer(
    #     cameras=cameras,
    #     raster_settings=raster_settings
    # )
    # rasterizer = rasterizer.to(device)
    
    # # Rasterize mesh - get face indices
    # fragments = rasterizer(mesh)

    # face_indices = fragments.pix_to_face.squeeze()  # (H, W)
    # z_buf = fragments.zbuf.squeeze()  # (H, W)

    # semantic_map = torch.zeros((h, w), dtype=torch.int32, device=device) - 1  # -1 for background
    # print("face_indices", face_indices.shape)
    # print("z_buf", z_buf.shape)
    # print("positive z_buf", (z_buf > 0).sum(), 'out of', z_buf.numel(), 'z min', z_buf.min(), 'z max', z_buf.max())
 
    # z_buf_to_save = z_buf.cpu().numpy()
    # z_buf_to_save_normalize = z_buf_to_save.copy()
    # z_buf_to_save_normalize = (z_buf_to_save - z_buf_to_save.min()) / (z_buf_to_save.max() - z_buf_to_save.min())
    # z_buf_to_save_normalize = (255 * z_buf_to_save_normalize).astype(np.uint8)
    # z_buf_to_save_normalize = cv2.applyColorMap(z_buf_to_save_normalize, cv2.COLORMAP_JET)
    # z_buf_to_save_normalize = Image.fromarray(z_buf_to_save_normalize)
    # z_buf_to_save_normalize.save('./z_buf.png')

    # raise ValueError("debug depth", z_buf.shape)

    # for h_idx in range(image_size[0]):
    #     for w_idx in range(image_size[1]):
    #         if face_indices[h_idx, w_idx] != -1:
    #             vertices_semantics_label = []
    #             for i in range(3):
    #                 vertices_semantics_label.append(semantics[faces[face_indices[h_idx, w_idx], i]])
    #             # max voting, if 3 vertices are different, using the first one
    #             unique_labels, unique_counts = np.unique(vertices_semantics_label, return_counts=True)
    #             max_count_index = np.argmax(unique_counts)
    #             semantic_map[h_idx, w_idx] = unique_labels[max_count_index]
    # return semantic_map  
    # # Project vertices to 2D
    # verts_2d = cameras.transform_points(verts.unsqueeze(0))  # (1, V, 3)
    # verts_2d = verts_2d.squeeze(0)[:, :2]  # (V, 2)
    
    # # Convert to pixel coordinates
    # h, w = image_size
    # pixel_coords = torch.zeros_like(verts_2d)
    # pixel_coords[:, 0] = (verts_2d[:, 0] + 1) * 0.5 * w  # x coord
    # pixel_coords[:, 1] = (verts_2d[:, 1] + 1) * 0.5 * h  # y coord
    
    # # Determine vertex visibility (any face using this vertex is visible)
    # visible_faces = face_indices[face_indices != -1].unique()
    # visible_verts = torch.zeros(len(verts), dtype=torch.bool, device=device)
    # if len(visible_faces) > 0:
    #     visible_verts = torch.any(faces[visible_faces].reshape(-1).unsqueeze(0) == 
    #                             torch.arange(len(verts), device=device).unsqueeze(1), dim=1)
        
    # print("number of visible vertices", visible_verts.sum())

    # semantic_map = torch.zeros((h, w), dtype=torch.int32, device=device) - 1  # -1 for background
    # depth_buffer = torch.full((h, w), float('inf'), device=device)  # Initialize depth buffer
    # for i in range(len(verts)):
    #     if visible_verts[i]:
    #         x, y = pixel_coords[i].long()
    #         if 0 <= x < w and 0 <= y < h:
    #             depth = fragments.zbuf.squeeze()[y, x]
    #             if depth < depth_buffer[y, x]:
    #                 depth_buffer[y, x] = depth
    #                 semantic_map[y, x] = semantics[i]
    
    # return {
    #     'face_indices': face_indices.cpu(),  # (H, W) with face indices
    #     'vertex_pixel_coords': pixel_coords.cpu(),  # (V, 2) pixel coordinates
    #     'vertex_visibility': visible_verts.cpu(),  # (V,) visibility mask
    #     'depth_map': fragments.zbuf.squeeze().cpu(),  # (H, W) depth values
    #     "semantic_map": semantic_map, # (H, W) semantic labels
    # }




# def project_mesh_open3d(verts, faces, semantics, image_size, intrinsic, extrinsic, raycast_samples=5):
#     """
#     Enhanced mesh projection with accurate visibility checking via ray-casting
    
#     Args:
#         verts: (V, 3) numpy array of vertices
#         faces: (F, 3) numpy array of faces
#         semantics: (V,) numpy array of vertex semantic labels
#         image_size: (width, height) tuple
#         intrinsic: o3d.camera.PinholeCameraIntrinsic
#         extrinsic: (4, 4) world-to-camera matrix
#         raycast_samples: Number of ray samples per vertex (default=5)
    
#     Returns:
#         - depth_image: (H, W) depth map
#         - vertex_pixel_coords: (V, 2) pixel coordinates
#         - vertex_visibility: (V,) bool array (True=visible)
#     """
#     # Create mesh and scene
#     mesh = o3d.geometry.TriangleMesh()
#     mesh.vertices = o3d.utility.Vector3dVector(verts)
#     mesh.triangles = o3d.utility.Vector3iVector(faces)
#     mesh.compute_vertex_normals()
    
#     # Setup camera
#     width, height = image_size
#     camera = o3d.camera.PinholeCameraParameters()
#     camera.intrinsic = intrinsic
#     camera.extrinsic = extrinsic  # World-to-camera
    
#     # Create visualizer for offscreen rendering
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(width=width, height=height, visible=False)
#     vis.add_geometry(mesh)
    
#     # Render depth
#     depth = vis.capture_depth_float_buffer(do_render=True)
#     vis.destroy_window()
    
#     # Convert to numpy
#     depth_image = np.asarray(depth)


#     # Project vertices to 2D
#     verts_hom = np.hstack((verts, np.ones((len(verts), 1))))
#     verts_cam = (extrinsic @ verts_hom.T).T[:, :3]
#     verts_2d = (intrinsic.intrinsic_matrix @ verts_cam.T).T
#     verts_2d = verts_2d[:, :2] / verts_2d[:, 2:]
    
#     # Convert to pixel coordinates (Open3D origin is top-left)
#     width, height = image_size
#     pixel_coords = np.zeros_like(verts_2d)
#     pixel_coords[:, 0] = verts_2d[:, 0]
#     pixel_coords[:, 1] = verts_2d[:, 1]
    
#     # --------------------------
#     # Enhanced Visibility Check
#     # --------------------------
#     vertex_visibility = np.zeros(len(verts), dtype=bool)
    
#     # Method 1: Fast but approximate (using depth buffer)
#     # depth_image = render_depth_map(mesh, intrinsic, extrinsic, image_size)



#     for i, (u, v) in enumerate(pixel_coords):
#         if 0 <= u < width and 0 <= v < height:
#             # Compare vertex depth to depth buffer with tolerance
#             vertex_depth = verts_cam[i, 2]
#             buffer_depth = depth_image[int(v), int(u)]
#             vertex_visibility[i] = abs(vertex_depth - buffer_depth) < 0.05  # 5cm tolerance
    
#     semantic_image = np.zeros((height, width), dtype=np.int32) - 1  # -1 for background
#     for i, (u, v) in enumerate(pixel_coords):
#         if vertex_visibility[i]:  # Only consider visible vertices
#             if 0 <= u < width and 0 <= v < height:
#                 semantic_image[int(v), int(u)] = semantics[i]
#     return semantic_image
#     # Method 2: Accurate but slower (ray-casting)
#     # if raycast_samples > 0:
#     #     visible_count = np.zeros(len(verts))
#     #     for i in tqdm(range(len(verts)), desc="Raycasting"):
#     #         if not vertex_visibility[i]:  # Only check uncertain vertices
#     #             u, v = pixel_coords[i]
#     #             if 0 <= u < width and 0 <= v < height:
#     #                 # Cast multiple rays around the vertex
#     #                 for _ in range(raycast_samples):
#     #                     # Jitter ray direction slightly
#     #                     ray_u = u + np.random.uniform(-0.5, 0.5)
#     #                     ray_v = v + np.random.uniform(-0.5, 0.5)
                        
#     #                     # Create ray (origin at camera, direction to vertex)
#     #                     ray_dir = verts_cam[i] / np.linalg.norm(verts_cam[i])
#     #                     ray = o3d.core.Tensor([[*ray_dir, 0]], dtype=o3d.core.Dtype.Float32)
                        
#     #                     # Cast ray
#     #                     ans = scene.cast_rays(ray)
#     #                     if ans['t_hit'].numpy()[0] < np.inf:
#     #                         visible_count[i] += 1
        
#     #     # Mark vertices hit by at least 50% of rays
#     #     vertex_visibility = visible_count >= (raycast_samples / 2)
    
#     # return {
#     #     'depth_image': depth_image,
#     #     'vertex_pixel_coords': pixel_coords,
#     #     'vertex_visibility': vertex_visibility,
#     #     'verts_camera_space': verts_cam  # Useful for debugging
#     # }



def fill_with_nearest_neighbor(arr):
    """Replace -1 values with nearest non -1 neighbor"""
    mask = (arr == -1)
    if not np.any(mask):
        return arr
    
    # Get indices of non -1 values
    non_empty_indices = np.argwhere(~mask)
    
    # Get indices of -1 values
    empty_indices = np.argwhere(mask)
    
    # Find closest non -1 index for each -1
    distances, closest_indices = distance_transform_edt(
        mask, 
        return_distances=True, 
        return_indices=True
    )
    
    # Use closest indices to fill -1 values
    filled = arr.copy()
    filled[mask] = filled[tuple(closest_indices[:, mask])]
    
    return filled

def project_points_to_2d(points_3d, labels, world_to_camera, camera_matrix, dist_coeffs, image_size):
    """
    Project 3D points to 2D using OpenCV's projection function
    
    Args:
        points_3d: Nx3 numpy array of 3D points in world coordinates
        labels: Nx1 numpy array of semantic labels
        world_to_camera: 4x4 transformation matrix
        camera_matrix: 3x3 intrinsic matrix
        dist_coeffs: distortion coefficients (usually just zeros for modern cameras)
        image_size: (width, height) tuple
    
    Returns:
        points_2d: Mx2 array of 2D coordinates (only points in front of camera)
        visible_labels: Mx1 array of corresponding labels
        depth: Mx1 array of depth values
    """
    # Transform points to camera coordinates
    points_cam = cv2.transform(points_3d.reshape(-1, 1, 3), world_to_camera[:3]).reshape(-1, 3)
    
    # Filter points behind camera (z < 0)
    in_front = points_cam[:, 2] > 0
    points_cam = points_cam[in_front]
    visible_labels = labels[in_front]
    
    # Project to 2D using OpenCV's projectPoints
    points_2d, _ = cv2.projectPoints(
        points_cam,
        np.zeros(3),  # rvec (zero rotation)
        np.zeros(3),  # tvec (zero translation)
        camera_matrix,
        dist_coeffs
    )
    points_2d = points_2d.reshape(-1, 2)
    
    # Filter points outside image bounds
    width, height = image_size
    in_bounds = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < width) & \
                (points_2d[:, 1] >= 0) & (points_2d[:, 1] < height)
    points_2d = points_2d[in_bounds]
    visible_labels = visible_labels[in_bounds]
    depths = points_cam[in_bounds, 2]
    
    # Initialize output buffers
    seg_image = np.zeros((height, width), dtype=np.int32)
    seg_image = seg_image - 1 # -1 for ignore index
    depth_buffer = np.full((height, width), np.inf)

    # Process each point with depth test
    points_2d_rounded = np.round(points_2d).astype(int)
    for (x, y), label, depth in zip(points_2d_rounded, visible_labels, depths):
        if 0 <= x < width and 0 <= y < height:
            # Only update if this point is closer than what's already there
            if depth < depth_buffer[y, x]:
                depth_buffer[y, x] = depth
                seg_image[y, x] = label

    # return points_2d, visible_labels, depth
    return seg_image


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def get_argparse():
    parser = argparse.ArgumentParser(description="Test a model")
    parser.add_argument(
        "--dataset_root", type=str, default='/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/', help="Path to the dataset root"
    )
    parser.add_argument(
        "--split_path",
        type=str,
        default='/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/splits/nvs_sem_val.txt'
    )
    args = parser.parse_args()
    return args

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



# python test_ply_3d.py --gs_path /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/neurips_2025/LangSplat_Qi/output/27dd4da69e_1/chkpnt30000.pth --gt_path_root /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/GS_Transformer/data/scannet_full/data/27dd4da69e/scans


if __name__ == '__main__':
    args = get_argparse()
    split_path = args.split_path 

    val_split = np.loadtxt(split_path, dtype=str)
    # print("val_split", val_split)
    val_split = sorted(val_split)[:1]
    for val_i in tqdm(val_split):
        val_data_path = os.path.join(args.dataset_root, 'data',val_i)
        selected_json = os.path.join(val_data_path, 'dslr', 'nerfstudio', 'lang_feat_selected_imgs.json')
        with open(selected_json, 'r') as f:
            selected_data_list = json.load(f)
        selected_frames = selected_data_list['frames']
        selected_imgs_list = [frame_i['file_path'] for frame_i in selected_frames]
        # borrow from nerfstuido
        # open the mesh

        segment_class_names = np.loadtxt(
            Path(args.dataset_root) / "metadata" / "semantic_benchmark" / "top100.txt",
            dtype=str,
            delimiter=".",  # dummy delimiter to replace " "
        )

        gt_mesh_path = os.path.join(val_data_path, 'scans', "mesh_aligned_0.05.ply")
        gt_segs_path = os.path.join(val_data_path, 'scans',  "segments.json")
        gt_anno_path = os.path.join(val_data_path, 'scans',  "segments_anno.json")

        # semantic_mesh = o3d.io.read_triangle_mesh(str(gt_semantic_mesh_path))
        # import trimesh 
        # semantic_mesh = trimesh.load(str(gt_semantic_mesh_path))
        # print("semantic_mesh", semantic_mesh) 

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

        semantic_gt = semantic_gt[:, 0]
        
        contents = selected_data_list
        org_height = contents["h"]
        org_width = contents["w"]
        focal_len_x = contents["fl_x"]
        focal_len_y = contents["fl_y"]
        cx = contents["cx"] * 2 
        cy = contents["cy"] * 2 
        fovx = focal2fov(focal_len_x, cx)
        fovy = focal2fov(focal_len_y, cy)

        FovY = fovy 
        FovX = fovx
        frames = contents["frames"]

        # coord = coord - coord.min(axis=0)
        verts = torch.tensor(coord, dtype=torch.float32).cuda()
        print("x min", verts[:, 0].min(), "x max", verts[:, 0].max())
        print("y min", verts[:, 1].min(), "y max", verts[:, 1].max())
        print("z min", verts[:, 2].min(), "z max", verts[:, 2].max())

        faces = torch.tensor(mesh.triangles, dtype=torch.int32).cuda()

        points_3d = np.array(coord)
        labels = np.array(semantic_gt)
        sparse_save_path = os.path.join(val_data_path, 'dslr', 'segmentation_2d_sparse')
        dense_save_path = os.path.join(val_data_path, 'dslr', 'segmentation_2d_dense')
        os.makedirs(sparse_save_path, exist_ok=True)
        os.makedirs(dense_save_path, exist_ok=True)


        for idx, frame in enumerate(frames):

            cam_name = frame["file_path"]

        #     # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            #  +X axis points left, the +Y is up and +Z into the screen (i.e. away from the observer). In that case a +90 CCW about the Z axis of a vector in the direction [1, 0, 0] would give [0, -1, 0] which is what you observed.
            resize = [876, 584]
            # resize_camera_matrix = np.array([
            #     [focal_len_x * resize[0] / org_width, 0, resize[0]//2, 0],
            #     [0, focal_len_y * resize[1] / org_height, resize[1] //2, 0],
            #     [0, 0, 1, 0],
            #     [0, 0, 0, 1]
            # ], dtype=float)


            applied_transform = np.array([
                [0,  1,  0,  0],
                [1,  0,  0,  0],
                [0,  0, -1,  0],
                [0,  0,  0,  1],
            ], dtype=float)
            c2w = np.dot(applied_transform, c2w)
            #     # get the world-to-camera transform and set R, T

            w2c = np.linalg.inv(c2w)
            w2c[1:3] *= -1
            opencv_w2c = w2c 
            opencv_c2w = np.linalg.inv(opencv_w2c)
            # flip_xy = np.array([
            #     [-1,  0,  0, 0],
            #     [0, -1,  0, 0],  # Flip Y
            #     [0,  0, 1, 0],  # Flip Z
            #     [0,  0,  0, 1]
            # ])
            pytorch3d_c2w = opencv_to_pytorch3d(opencv_c2w)
            # pytorch3d_c2w =  opencv_c2w @ flip_xy
            pytorch3d_w2c = np.linalg.inv(pytorch3d_c2w)
            # R = pytorch3d_w2c[:3,:3]  
            # T = pytorch3d_c2w[:3, 3]
            R = np.eye(3)
            T = np.array([4,4,1])

            resize = [876, 584]
            img_size = [resize[1], resize[0]]

            # s1 = focal_len_x * resize[0] / org_width
            # s2 = focal_len_y * resize[1] / org_height
            # w1 = resize[0] // 2
            # h1 = resize[1] // 2

            # Projection matrix
            # K = [
            #         [s1,   0,   w1,   0],
            #         [0,   s2,   h1,   0],
            #         [0,    0,   f1,  f2],
            #         [0,    0,    1,   0],
            # ]
            print("fovx", fovx, "fovy", fovy)
            camera_params = {
                'R': torch.tensor(R, dtype=torch.float32).cuda(),
                'T': torch.tensor(T, dtype=torch.float32).cuda(),
                'focal_length': torch.tensor([focal_len_x * resize[0] / org_width, focal_len_y * resize[1] / org_height], dtype=torch.float32).cuda(),
                'principal_point': torch.tensor([resize[0]//2, resize[1]//2], dtype=torch.float32).cuda(),
                'fovx': torch.tensor(fovx, dtype=torch.float32).cuda(),
                'fovy': torch.tensor(fovy, dtype=torch.float32).cuda(),
                'w': torch.tensor(resize[0], dtype=torch.float32).cuda(),
                'h': torch.tensor(resize[1], dtype=torch.float32).cuda(),
                'focal_len_x': torch.tensor(focal_len_x * resize[0] / org_width, dtype=torch.float32).cuda(),
                'focal_len_y': torch.tensor(focal_len_y * resize[1] / org_height, dtype=torch.float32).cuda(),
                # 'K': torch.tensor(resize_camera_matrix, dtype=torch.float32).cuda()
            }
            opencv_w2c = torch.tensor(opencv_w2c, dtype=torch.float32).cuda()
            semantic_gt=  torch.tensor(semantic_gt, dtype=torch.int32).cuda()
            face_vertices_camera, face_vertices_image, _ = prepare_vertices_ndc(
                    verts, faces, camera_transform=opencv_w2c, camera_params=camera_params, device='cuda')
            
            depth, seg_image_sparse = render_depth_plus_pytorch3d(face_vertices_image, face_vertices_camera[:, :, :, -1], semantic_gt, camera_params, device='cuda')
            
            # seg_image_sparse = project_mesh_to_2d(verts, faces, semantic_gt, img_size, camera_params,device='cuda')
            
            # seg_image_sparse = project_mesh_open3d(verts=verts.cpu().numpy(), faces=faces.cpu().numpy(), semantics=semantic_gt, image_size=(resize[0], resize[1]), intrinsic=resize_camera_matrix_intrinsic, extrinsic=w2c, raycast_samples=5)
            seg_image_sparse = seg_image_sparse.cpu().numpy()
            seg_image_sparse_to_save = seg_image_sparse.copy()

            seg_image_sparse_non_ignore = seg_image_sparse!= -1
            print("number of points in seg_image_sparse", seg_image_sparse_non_ignore.sum())

            print("seg_image_sparse_to_save", seg_image_sparse_to_save.min(), seg_image_sparse_to_save.max())

            # we save the sparse, desnify here
            seg_image_dense = fill_with_nearest_neighbor(seg_image_sparse)

            seg_image_sparse_save_path = os.path.join(val_data_path, 'dslr', 'segmentation_2d_sparse', cam_name.split('.')[0] + '.png')
            seg_image_dense_save_path = os.path.join(val_data_path, 'dslr', 'segmentation_2d_dense', cam_name.split('.')[0] + '.png')

            seg_image_dense_to_save = seg_image_dense.copy()
            # save to png for visualization
            seg_image_sparse_to_save = seg_image_sparse_to_save.astype(np.uint8)
            seg_image_sparse_to_save[seg_image_sparse_to_save == -1] = 0
            seg_image_dense_to_save = seg_image_dense_to_save.astype(np.uint8)
            seg_image_dense_to_save[seg_image_dense_to_save == -1] = 0
            # colormap for better visualization
            seg_image_sparse_to_save = cv2.applyColorMap(seg_image_sparse_to_save, cv2.COLORMAP_JET)
            seg_image_dense_to_save = cv2.applyColorMap(seg_image_dense_to_save, cv2.COLORMAP_JET)

            seg_image_sparse_to_save = Image.fromarray(seg_image_sparse_to_save)
            seg_image_dense_to_save = Image.fromarray(seg_image_dense_to_save)
            seg_image_sparse_to_save.save(seg_image_sparse_save_path)
            seg_image_dense_to_save.save(seg_image_dense_save_path)
            print("seg_image_sparse_save_path", seg_image_sparse_save_path)


            # save to npy for future use,
            np.save(seg_image_sparse_save_path.replace('.png', '.npy'), seg_image_sparse_to_save)
            np.save(seg_image_dense_save_path.replace('.png', '.npy'), seg_image_dense)


            raise ValueError("mesh")
    

        # dummyp projection

        # # # raise ValueError("Frames: ", frames)
        # for idx, frame in enumerate(frames):

        #     cam_name = frame["file_path"]

        # #     # NeRF 'transform_matrix' is a camera-to-world transform
        #     c2w = np.array(frame["transform_matrix"])
        #     # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        #     applied_transform = np.array([
        #         [0,  1,  0,  0],
        #         [1,  0,  0,  0],
        #         [0,  0, -1,  0],
        #         [0,  0,  0,  1],
        #     ], dtype=float)
        #     c2w = np.dot(applied_transform, c2w)
        #     #     # get the world-to-camera transform and set R, T

        #     w2c = np.linalg.inv(c2w)
        #     w2c[1:3] *= -1
        #     world_to_camera = w2c 

        #     # camera_matrix = np.array([
        #     #     [focal_len_x, 0, cx],
        #     #     [0, focal_len_y, cy],
        #     #     [0, 0, 1]
        #     # ], dtype=float)
        #     resize = [876, 584]
        #     resize_camera_matrix = np.array([
        #         [focal_len_x * resize[0] / org_width, 0, resize[0]//2],
        #         [0, focal_len_y * resize[1] / org_height, resize[1] //2],
        #         [0, 0, 1]
        #     ], dtype=float)
        #     dist_coeffs = np.zeros((4, 1), dtype=float)
        #     sparse_size = resize

        #     seg_image_sparse = project_points_to_2d(
        #         points_3d, labels, world_to_camera, resize_camera_matrix, dist_coeffs, sparse_size)
            
        #     # print("seg_image_sparse", seg_image_sparse.shape)
        #     # print("unqiue seg_image_sparse", np.unique(seg_image_sparse))
        #     # seg_image_sparse_non_ignore = seg_image_sparse!= -1
        #     # print("number of points in seg_image_sparse", seg_image_sparse_non_ignore.sum())
        #     seg_image_sparse_to_save = seg_image_sparse.copy()

        #     # we save the sparse, desnify here
        #     seg_image_dense = fill_with_nearest_neighbor(seg_image_sparse)

        #     seg_image_sparse_save_path = os.path.join(val_data_path, 'dslr', 'segmentation_2d_sparse', cam_name.split('.')[0] + '.png')
        #     seg_image_dense_save_path = os.path.join(val_data_path, 'dslr', 'segmentation_2d_dense', cam_name.split('.')[0] + '.png')

        #     seg_image_dense_to_save = seg_image_dense.copy()
        #     # save to png for visualization
        #     seg_image_sparse_to_save = seg_image_sparse_to_save.astype(np.uint8)
        #     seg_image_sparse_to_save[seg_image_sparse_to_save == -1] = 0
        #     seg_image_dense_to_save = seg_image_dense_to_save.astype(np.uint8)
        #     seg_image_dense_to_save[seg_image_dense_to_save == -1] = 0
        #     # colormap for better visualization
        #     seg_image_sparse_to_save = cv2.applyColorMap(seg_image_sparse_to_save, cv2.COLORMAP_JET)
        #     seg_image_dense_to_save = cv2.applyColorMap(seg_image_dense_to_save, cv2.COLORMAP_JET)

        #     seg_image_sparse_to_save = Image.fromarray(seg_image_sparse_to_save)
        #     seg_image_dense_to_save = Image.fromarray(seg_image_dense_to_save)
        #     seg_image_sparse_to_save.save(seg_image_sparse_save_path)
        #     seg_image_dense_to_save.save(seg_image_dense_save_path)


        #     # save to npy for future use,
        #     np.save(seg_image_sparse_save_path.replace('.png', '.npy'), seg_image_sparse_to_save)
        #     np.save(seg_image_dense_save_path.replace('.png', '.npy'), seg_image_dense)

        #     raise ValueError("seg_image_sparse_save_path", seg_image_sparse_save_path)
        
        #     w2c[1:3] *= -1
        #     R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        #     T = w2c[:3, 3]
        #     image_path = os.path.join(image_path, cam_name)
        #     image_name = Path(cam_name).stem
    

        #     image = Image.open(image_path)
        #     # resize
        #     resize = [584, 876]
        #     resize_img = (
        #         (resize[1], resize[0])
        #         if resize[1] > resize[0]
        #         else (resize[0], resize[1])
        #     )
        #     # image = image.resize(resize_img, Image.Resampling.LANCZOS)
        #     image = image.resize(resize_img, Image.LANCZOS)
