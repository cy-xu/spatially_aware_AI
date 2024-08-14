import argparse
import glob
import json
import os

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import pandas as pd
import skimage.measure
import torch
import tqdm
import trimesh
import yaml


control_objects = [
    "airplane",
    "bag",
    "bed",
    "bedclothes",
    "bench",
    "bicycle",
    "bird",
    "boat",
    "book",
    "bottle",
    "building",
    "bus",
    "cabinet",
    "car",
    "cat",
    "ceiling",
    "chair",
    "cloth",
    "computer",
    "cow",
    "cup",
    "curtain",
    "dog",
    "door",
    "fence",
    "floor",
    "flower",
    "food",
    "grass",
    "ground",
    "horse",
    "keyboard",
    "light",
    "motorbike",
    "mountain",
    "mouse",
    "person",
    "plate",
    "platform",
    "potted plant",
    "road",
    "rock",
    "sheep",
    "shelves",
    "sidewalk",
    "sign",
    "sky",
    "snow",
    "sofa",
    "table",
    "track",
    "train",
    "tree",
    "truck",
    "tv monitor",
    "wall",
    "water",
    "window",
    "wood",
    "sharp edges",
    "computer screen",
    "rug",
    "sharp corners",
]


class LERFDataset(torch.utils.data.Dataset):
    def __init__(self, scan_dir):
        rgb_imgfiles = sorted(glob.glob(os.path.join(scan_dir, "images/*.jpg")))
        depth_imgfiles = sorted(
            glob.glob(os.path.join(scan_dir, "depth_simplerecon/*.png"))
        )

        with open(os.path.join(scan_dir, "transforms.json"), "r") as f:
            transforms = json.load(f)

        self.poses = []
        transforms["frames"] = sorted(
            transforms["frames"], key=lambda frame: frame["file_path"]
        )
        for frame in transforms["frames"]:
            pose = torch.tensor(frame["transform_matrix"])

            if "applied_transform" in transforms:
                t = torch.eye(4)
                t[:3] = torch.tensor(transforms["applied_transform"])
                pose = t.inverse() @ pose

            pose[:3, 1] *= -1
            pose[:3, 2] *= -1

            frame_id = os.path.basename(frame["file_path"]).split(".")[0]
            self.poses.append({"frame_id": frame_id, "pose": pose})

        self.rgb_imgfiles = {}
        for f in rgb_imgfiles:
            frame_id = os.path.basename(f).split(".")[0]
            self.rgb_imgfiles[frame_id] = f

        self.depth_imgfiles = {}
        for f in depth_imgfiles:
            frame_id = os.path.basename(f).split(".")[0]
            self.depth_imgfiles[frame_id] = f

        native_imheight, native_imwidth, _ = cv2.imread(
            list(self.rgb_imgfiles.values())[0]
        ).shape
        depth_imheight, depth_imwidth = cv2.imread(
            list(self.depth_imgfiles.values())[0], cv2.IMREAD_ANYDEPTH
        ).shape

        self.imwidth = depth_imwidth
        self.imheight = depth_imheight

        self.K = {}

        if "fl_x" in transforms:
            fx = transforms["fl_x"]
            fy = transforms["fl_y"]
            cx = transforms["cx"]
            cy = transforms["cy"]
            K = torch.tensor(
                [
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1],
                ]
            ).float()
            K[0] *= self.imwidth / native_imwidth
            K[1] *= self.imheight / native_imheight
            for frame in transforms["frames"]:
                frame_id = os.path.basename(frame["file_path"]).split(".")[0]
                self.K[frame_id] = K

        else:
            for frame in transforms["frames"]:
                frame_id = os.path.basename(frame["file_path"]).split(".")[0]
                fx = frame["fl_x"]
                fy = frame["fl_y"]
                cx = frame["cx"]
                cy = frame["cy"]
                K = torch.tensor(
                    [
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1],
                    ]
                ).float()
                K[0] *= self.imwidth / native_imwidth
                K[1] *= self.imheight / native_imheight
                self.K[frame_id] = K

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        frame_id = self.poses[idx]["frame_id"]
        pose = self.poses[idx]["pose"]

        rgb_img = cv2.cvtColor(
            cv2.imread(self.rgb_imgfiles[frame_id]), cv2.COLOR_BGR2RGB
        )
        rgb_img = cv2.resize(
            rgb_img, (self.imwidth, self.imheight), None, 0, 0, cv2.INTER_AREA
        )
        rgb_img = torch.from_numpy(rgb_img).float() / 255

        depth_img = cv2.imread(self.depth_imgfiles[frame_id], cv2.IMREAD_ANYDEPTH)
        depth_img = torch.from_numpy(depth_img.astype(np.float32)) / 1000

        return rgb_img, depth_img, pose, self.K[frame_id], idx


class ScanNetDataset(torch.utils.data.Dataset):
    def __init__(self, scan_dir):
        self.dataset_name = "scannet"

        self.rgb_imgfiles = sorted(
            glob.glob(os.path.join(scan_dir, "color/*.jpg")),
            key=lambda f: int(os.path.basename(f).split(".")[0]),
        )
        self.depth_imgfiles = sorted(
            glob.glob(os.path.join(scan_dir, "depth/*.png")),
            key=lambda f: int(os.path.basename(f).split(".")[0]),
        )
        posefiles = sorted(
            glob.glob(os.path.join(scan_dir, "pose/*.txt")),
            key=lambda f: int(os.path.basename(f).split(".")[0]),
        )
        K_file = os.path.join(scan_dir, "intrinsic/intrinsic_depth.txt")

        # K is the depth camera intrinsic matrix (only keep the 3x3 matrix)
        # which transforms 3D points in the camera coordinate system to 2D points on the image plane.
        self.K = torch.from_numpy(np.loadtxt(K_file)).float()[:3, :3]
        poses = np.stack([np.loadtxt(f) for f in posefiles], axis=0)
        good_pose = ~np.any(np.isinf(poses), axis=(1, 2))
        self.poses = torch.from_numpy(poses[good_pose]).float()

        self.rgb_imgfiles = np.array(self.rgb_imgfiles)[good_pose]
        self.depth_imgfiles = np.array(self.depth_imgfiles)[good_pose]

        # hardcoded image size to match depth image size
        self.imwidth = 640
        self.imheight = 480

        # calculate pose distance between two frames, select useful keyframes
        kf_idx = [0]
        last_kf_pose = self.poses[0]
        for i in range(1, len(self.poses)):
            tdist = torch.norm(self.poses[i, :3, 3] - last_kf_pose[:3, 3])
            if tdist > 0.1:
                kf_idx.append(i)
                last_kf_pose = poses[i]
        kf_idx = np.array(kf_idx)

        self.kf_idx = kf_idx
        self.poses = self.poses[kf_idx]
        self.rgb_imgfiles = self.rgb_imgfiles[kf_idx]
        self.depth_imgfiles = self.depth_imgfiles[kf_idx]

    def __len__(self):
        return len(self.depth_imgfiles)

    def __getitem__(self, i):
        rgb_img = cv2.cvtColor(cv2.imread(self.rgb_imgfiles[i]), cv2.COLOR_BGR2RGB)
        rgb_img = torch.from_numpy(rgb_img).float() / 255
        rgb_img = torch.nn.functional.interpolate(
            rgb_img.permute(2, 0, 1)[None],
            size=(self.imheight, self.imwidth),
            mode="bilinear",
            align_corners=False,
        )[0].permute(1, 2, 0)

        depth_img = cv2.imread(self.depth_imgfiles[i], cv2.IMREAD_ANYDEPTH)
        depth_img = torch.from_numpy(depth_img.astype(np.float32)) / 1000

        return rgb_img, depth_img, self.poses[i], self.K, self.kf_idx[i]


class iPhone3DScannerDataset(torch.utils.data.Dataset):
    def __init__(self, scan_dir, views_limit=0):
        self.dataset_name = "iphone3dscanner"

        self.rgb_imgfiles = sorted(glob.glob(os.path.join(scan_dir, "frame_*.jpg")))
        if views_limit > 0:
            self.rgb_imgfiles = self.rgb_imgfiles[:views_limit]

        # debug
        # self.rgb_imgfiles = [self.rgb_imgfiles[0], self.rgb_imgfiles[-1]]

        # not rgb frames are collected, only collect the depth frames when a rgb frame is available
        self.pose_files = []
        self.depth_files = []
        # depth value in millimeters
        for img_file in self.rgb_imgfiles:
            depth_file = img_file.replace(".jpg", ".png")
            depth_file = depth_file.replace("frame_", "depth_")
            if os.path.exists(depth_file):
                self.depth_files.append(depth_file)

            pose_file = img_file.replace(".jpg", ".json")
            if os.path.exists(pose_file):
                self.pose_files.append(pose_file)

        assert len(self.rgb_imgfiles) == len(self.depth_files)
        assert len(self.rgb_imgfiles) == len(self.pose_files)

        # K_file = os.path.join(scan_dir, "intrinsic/intrinsic_depth.txt")

        # K is the depth camera intrinsic matrix (only keep the 3x3 matrix)
        # which transforms 3D points in the camera coordinate system to 2D points on the image plane.
        self.metadata = []

        for posefile in self.pose_files:
            metadata = {}
            with open(posefile, "r") as f:
                meta = json.load(f)
                intrinsics = np.array(meta["intrinsics"]).reshape(3, 3)
                pose = np.array(meta["cameraPoseARFrame"]).reshape(4, 4)
                projection_matrix = np.array(meta["projectionMatrix"]).reshape(4, 4)
                motion_quality = meta["motionQuality"]
                angular_velocity = meta["averageAngularVelocity"]

                metadata["intrinsics"] = intrinsics
                metadata["projection_matrix"] = projection_matrix
                metadata["motion_quality"] = motion_quality
                metadata["angular_velocity"] = angular_velocity

                # this app's camera coordinate is right-up-backward
                # clip fuion expects right-down-forward
                metadata["pose"] = pose
                metadata["pose"][:3, 1] *= -1
                metadata["pose"][:3, 2] *= -1

                self.metadata.append(metadata)

        # check invalid poses
        poses = np.stack([m["pose"] for m in self.metadata], axis=0)
        good_pose = ~np.any(np.isinf(poses), axis=(1, 2))
        self.poses = torch.from_numpy(poses[good_pose]).float()

        self.rgb_imgfiles = np.array(self.rgb_imgfiles)[good_pose]
        self.depth_files = np.array(self.depth_files)[good_pose]

        # hardcoded image size to match depth image size
        self.dep_h, self.dep_w = cv2.imread(
            self.depth_files[0], cv2.IMREAD_ANYDEPTH
        ).shape
        self.rgb_h, self.rgb_w, _ = cv2.imread(self.rgb_imgfiles[0]).shape

        # ensure video is filmed horizontally
        assert self.rgb_w > self.rgb_h

        # depth image original 256x192
        # rgb image original 1920x1440
        # resize to 1024x768
        self.imwidth = self.dep_w * 4
        self.imheight = self.dep_h * 4

        self.reflection_across_z = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32
        )

    def __len__(self):
        return len(self.depth_files)

    def __getitem__(self, i):
        # !!! if filmed vertically on iphone
        # 3d scanner output depth image is horizontal, rgb image could be vertical

        rgb_img = cv2.cvtColor(cv2.imread(self.rgb_imgfiles[i]), cv2.COLOR_BGR2RGB)
        # shrink the rgb image so we use cv2.INTER_AREA
        rgb_img = cv2.resize(
            rgb_img, (self.imwidth, self.imheight), None, 0, 0, cv2.INTER_AREA
        )
        rgb_img = torch.from_numpy(rgb_img).float() / 255

        depth_img = cv2.imread(self.depth_files[i], cv2.IMREAD_ANYDEPTH)
        # enlarge the depth image so we use cv2.INTER_CUBIC
        depth_img = cv2.resize(
            depth_img, (self.imwidth, self.imheight), None, 0, 0, cv2.INTER_CUBIC
        )
        depth_img = torch.from_numpy(depth_img.astype(np.float32)) / 1000

        pose = torch.from_numpy(self.metadata[i]["pose"]).float()
        # pose =  pose @ torch.from_numpy(self.reflection_across_x)

        # iphone provides K for each frame
        # K = torch.from_numpy(self.metadata[i]['intrinsics']).float()
        K = torch.from_numpy(self.metadata[i]["intrinsics"]).float()

        K[0] *= self.imwidth / self.rgb_w
        K[1] *= self.imheight / self.rgb_h

        return rgb_img, depth_img, pose, K, i


class HypersimDataset(torch.utils.data.Dataset):
    def __init__(self, scan_dir):
        self.dataset_name = "hypersim"

        # collect image files
        self.depth_imgfiles = sorted(
            glob.glob(
                f"{scan_dir}/images/scene_cam_00_geometry_hdf5/frame.*.depth_meters.hdf5"
            )
        )
        self.rgb_imgfiles = sorted(
            glob.glob(f"{scan_dir}/images/scene_cam_00_final_hdf5/frame.*.color.hdf5")
        )

        scan_name = os.path.basename(scan_dir)
        cam_csv = pd.read_csv(
            os.path.join(scan_dir, "../metadata_camera_parameters.csv")
        )
        cam_csv = cam_csv[cam_csv.scene_name == scan_name]

        self.imheight = int(np.round(cam_csv.settings_output_img_height.iloc[0]))
        self.imwidth = int(np.round(cam_csv.settings_output_img_width.iloc[0]))

        # collect camera poses and intrinsics
        metadata = pd.read_csv(f"{scan_dir}/_detail/metadata_scene.csv")
        meters_per_asset_unit = (
            metadata[metadata.parameter_name == "meters_per_asset_unit"]
        ).parameter_value.iloc[0]

        cam_pos_file = f"{scan_dir}/_detail/cam_00/camera_keyframe_positions.hdf5"
        cam_pos_dset = h5py.File(cam_pos_file)
        cam_pos = torch.from_numpy(cam_pos_dset["dataset"][:]).float()
        cam_pos *= meters_per_asset_unit

        cam_rot_file = f"{scan_dir}/_detail/cam_00/camera_keyframe_orientations.hdf5"
        cam_rot_dset = h5py.File(cam_rot_file)
        cam_rot = torch.from_numpy(cam_rot_dset["dataset"][:]).float()

        self.poses = torch.eye(4)[None].repeat(len(cam_pos), 1, 1)
        self.poses[:, :3, 3] = cam_pos
        self.poses[:, :3, :3] = cam_rot

        # Camera calibration matrix
        M = torch.tensor(
            [
                [
                    cam_csv.M_cam_from_uv_00.iloc[0],
                    cam_csv.M_cam_from_uv_01.iloc[0],
                    cam_csv.M_cam_from_uv_02.iloc[0],
                ],
                [
                    cam_csv.M_cam_from_uv_10.iloc[0],
                    cam_csv.M_cam_from_uv_11.iloc[0],
                    cam_csv.M_cam_from_uv_12.iloc[0],
                ],
                [
                    cam_csv.M_cam_from_uv_20.iloc[0],
                    cam_csv.M_cam_from_uv_21.iloc[0],
                    cam_csv.M_cam_from_uv_22.iloc[0],
                ],
            ],
            dtype=torch.float32,
        )

        # K is the intrinsic matrix (inversed?)
        self.K = self.M_to_K(M, self.imwidth, self.imheight)

        # pix_vecs converts a pixel coordinate to a ray direction
        pix_vecs = get_pix_vecs(self.imwidth, self.imheight, self.K[None])

        # TODO: self.dist_to_depth here means? Normalized by norm?
        self.dist_to_depth = 1 / pix_vecs.norm(dim=-1).view(self.imheight, self.imwidth)

    def __len__(self):
        return len(self.rgb_imgfiles)

    def __getitem__(self, i):
        with h5py.File(self.rgb_imgfiles[i], "r") as dset:
            rgb_img = dset["dataset"][:]
        rgb_img = torch.from_numpy(rgb_img.astype(np.float32)).clamp(0, 1)

        with h5py.File(self.depth_imgfiles[i], "r") as dset:
            depth_img = dset["dataset"][:]
        depth_img = torch.from_numpy(depth_img.astype(np.float32))
        depth_img *= self.dist_to_depth

        frame_idx = int(os.path.basename(self.depth_imgfiles[i]).split(".")[1])

        return rgb_img, depth_img, self.poses[frame_idx], self.K, i

    def M_to_K(self, M, imwidth, imheight):
        """Convert camera calibration matrix to intrinsic matrix"""
        u_min = -1.0
        u_max = 1.0
        v_min = -1.0
        v_max = 1.0
        half_du = 0.5 * (u_max - u_min) / imwidth
        half_dv = 0.5 * (v_max - v_min) / imheight

        fx = M[0, 0] * (2 * (u_max - half_du)) / (imwidth - 1)
        fy = M[1, 1] * (2 * (v_max - half_dv)) / (imheight - 1)

        cx = M[0, 0] * (u_min + half_du)
        cy = M[1, 1] * (v_min + half_dv)

        w0 = M[2, 0] * (2 * (u_max - half_du)) / (imwidth - 1)
        w1 = M[2, 1] * (2 * (v_max - half_dv)) / (imheight - 1)

        w2 = M[2, 0] * (u_min + half_du)
        w3 = M[2, 1] * (v_min + half_dv)

        K = torch.tensor(
            [[fx, 0, cx + M[0, 2]], [0, fy, cy + M[1, 2]], [w0, w1, M[2, 2] + w2 + w3]],
            dtype=torch.float32,
        )
        K[1] *= -1

        return K.inverse()


def get_pix_vecs(imwidth, imheight, K):
    device = K.device
    u = torch.arange(imwidth, dtype=torch.float32, device=device)
    v = torch.arange(imheight, dtype=torch.float32, device=device)
    uu, vv = torch.meshgrid(u, v, indexing="xy")
    uv = torch.stack((uu, vv, torch.ones_like(uu)), dim=-1).view(-1, 3)
    # invserse of intrinsic matrix K
    pix_vecs = (K.inverse() @ uv.T).transpose(1, 2)
    # pix_vecs is a tensor of shape (imheight * imwidth, 3)
    # TODO: its purpose is to convert a pixel coordinate to a ray direction
    return pix_vecs


def backproject_pcd(
    dataset, batch_size=1, num_workers=0, device="cpu", max_depth=torch.inf
):
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers
    )
    # downsampled sampling grid
    uv_size = 7
    u = torch.round(torch.linspace(0, dataset.imwidth - 1, uv_size)).long()
    v = torch.round(torch.linspace(0, dataset.imheight - 1, uv_size)).long()
    # uv is 7 by 7 grid of pixel coordinates

    # uv with full resolution
    # u = torch.linspace(0, dataset.imwidth - 1, dataset.imwidth).long()
    # v = torch.linspace(0, dataset.imheight - 1, dataset.imheight).long()

    uu, vv = torch.meshgrid(u, v, indexing="xy")
    uv = torch.stack((uu, vv), dim=-1).view(-1, 2)

    xyz = []
    rgb = []
    for view in tqdm.tqdm(loader, leave=False, desc="backproject pcd"):
        rgb_imgs, depth_imgs, poses, K, _ = view
        # poses is camera pose read from file (device's local coordinate?)
        # K is the intrinsic matrix (inversed?)
        batch_size = len(rgb_imgs)

        depth_imgs = depth_imgs.to(device)
        poses = poses.to(device)
        K = K.to(device)

        pix_vecs = get_pix_vecs(dataset.imwidth, dataset.imheight, K).to(device)
        # pix vecs projects a pixel coordinate to a ray direction, for each pixel
        pix_vecs = pix_vecs.view(len(pix_vecs), dataset.imheight, dataset.imwidth, 3)
        # reshape back to [1, imheight, imwidth, 3]
        pix_vecs = pix_vecs[:, uv[:, 1], uv[:, 0]]
        # sample pix_vecs at uv coordinates

        # sample 49 points from depth image as well
        depth = depth_imgs[:, uv[:, 1], uv[:, 0]]
        # tilda flips the true/false
        valid = (~depth.isnan()) & (depth > 0) & (depth < max_depth)

        xyz_cam = pix_vecs * depth[:, :, None]
        # pixel vectors multiplied by pixel depth
        # xyz_cam is the point cloud in camera coordiantes

        # batch_xyz = (poses[:, :3, :3] @ xyz_cam.transpose(1, 2)) + poses[:, :3, 3, None]
        batch_xyz = (poses[:, :3, :3] @ xyz_cam.transpose(1, 2)) + poses[:, :3, 3, None]
        # multiply rotation matrix with xyz_cam, then add translation
        # pose is extrisic of the view, multiply pose with xyz_cam
        # batch_xyz is the point cloud in world coordinate
        batch_rgb = rgb_imgs[:, uv[:, 1], uv[:, 0]]

        batch_xyz_flat = batch_xyz.masked_select(valid[:, None]).view(3, -1).T
        batch_rgb_flat = batch_rgb[valid]

        xyz.append(batch_xyz_flat.cpu())
        rgb.append(batch_rgb_flat)

    xyz = torch.cat(xyz, axis=0)
    rgb = torch.cat(rgb, axis=0)
    return xyz, rgb


class ClipFusion(torch.nn.Module):
    def __init__(
        self,
        origin,
        voxel_size,
        nvox,
        trunc,
        scale_patches_by_depth,
        clip_model,
        clip_pretraining,
        clip_patch_size,
        clip_patch_stride,
    ):
        super().__init__()

        self.clip = Clip(clip_model, clip_pretraining)
        self.clip.requires_grad_(False)
        self.clip.eval()

        # minbound is the origin of the scene
        self.origin = origin
        self.voxel_size = voxel_size
        self.nvox = nvox
        self.trunc = trunc
        self.clip_patch_size = clip_patch_size
        self.clip_patch_stride = clip_patch_stride
        self.scale_patches_by_depth = scale_patches_by_depth

        self.n_clip_feats = self.clip.feature_dim

        self.register_buffer("tsdf", torch.zeros(torch.prod(nvox)))
        self.register_buffer("rgb", torch.zeros(torch.prod(nvox), 3))
        self.register_buffer(
            "clip_feat", torch.zeros(torch.prod(nvox), self.n_clip_feats)
        )
        self.register_buffer("weight", torch.zeros(torch.prod(nvox), dtype=torch.int32))
        self.register_buffer(
            "tsdf_weight", torch.zeros(torch.prod(nvox), dtype=torch.int32)
        )
        # Here is where I want to add more features to each voxel
        # instead of clip_feat only, each voxel has a dictionary of various features

        x = torch.arange(nvox[0])
        y = torch.arange(nvox[1])
        z = torch.arange(nvox[2])
        xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
        xyz_idx = torch.stack((xx, yy, zz), dim=-1).view(-1, 3)
        xyz_world = xyz_idx * self.voxel_size + self.origin

        # TODO: xyz_world is the world coordinate of each voxel
        self.register_buffer("xyz_world", xyz_world)

    def integrate(self, depth_imgs, rgb_imgs, poses, K):
        batch_size, imheight, imwidth, _ = rgb_imgs.shape
        device = rgb_imgs.device

        if self.scale_patches_by_depth:
            # this makes sense
            # TODO: not used at the moment
            clip_feat_img = self.clip.img_inference_tiled_depthscaled(
                rgb_imgs.permute(0, 3, 1, 2),
                depth_imgs,
                K,
                patch_stride=self.clip_patch_stride,
            )
        else:
            clip_feat_img = self.clip.img_inference_tiled(
                rgb_imgs.permute(0, 3, 1, 2),
                patch_size=self.clip_patch_size,
                patch_stride=self.clip_patch_stride,
            )

        # calculates the xyz coordinate of each voxel in camera coordinate
        xyz_cam = poses[:, :3, :3].transpose(1, 2) @ (
            self.xyz_world[None] - poses[:, None, :3, 3]
        ).transpose(1, 2)

        uvz = K @ xyz_cam
        z = uvz[:, 2]
        uv = uvz[:, :2] / z[:, None]

        grid = uv + 0.5
        grid /= torch.tensor([imwidth, imheight], device=device)[None, :, None]
        grid *= 2
        grid -= 1

        depth = torch.nn.functional.grid_sample(
            depth_imgs[:, None],
            grid.transpose(1, 2)[:, None],
            mode="nearest",
            align_corners=False,
        )[:, 0, 0]

        # TODO: sdf is the signed distance of each voxel to the surface
        sdf = (depth - z) / self.trunc
        # tsdf is the truncated sdf
        tsdf = sdf.clamp(-1, 1)

        _valid = (grid.abs() <= 1).all(dim=1) & (z > 0)

        # here we truncate the sdf to one voxel size, too much?
        # _valid = (grid.abs() <= self.trunc).all(dim=1) & (z > 0)

        valid = _valid & (sdf.abs() <= 1)
        tsdf_valid = _valid & (sdf > -1)

        batch_tsdf_weight = tsdf_valid.sum(dim=0)
        batch_tsdf_valid = batch_tsdf_weight > 0
        tsdf.masked_fill_(~tsdf_valid, 0)
        batch_tsdf = tsdf.sum(dim=0)

        # TODO: what is the weight for?
        new_weight = self.tsdf_weight + batch_tsdf_weight.to(self.tsdf_weight.dtype)

        a = new_weight[batch_tsdf_valid]
        b = (self.tsdf_weight / new_weight)[batch_tsdf_valid]

        self.tsdf[batch_tsdf_valid] = (
            batch_tsdf[batch_tsdf_valid] / a + self.tsdf[batch_tsdf_valid] * b
        )
        self.tsdf_weight.copy_(new_weight)

        rgb_imgs = rgb_imgs.permute(0, 3, 1, 2)
        for i in range(batch_size):
            grid_valid = grid[i].T[valid[i]]

            rgb = torch.nn.functional.grid_sample(
                rgb_imgs[i, None],
                grid_valid[None, None],
                mode="nearest",
                align_corners=False,
            )[0, :, 0]

            clip_feat = torch.nn.functional.grid_sample(
                clip_feat_img[i, None, : self.n_clip_feats],
                grid_valid[None, None],
                mode="bilinear",
                align_corners=False,
            )[0, :, 0]

            new_weight = self.weight + valid[i].to(self.weight.dtype)
            a = 1 / new_weight[valid[i], None]
            b = self.weight[valid[i], None] * a
            self.rgb[valid[i]] = rgb.T * a + self.rgb[valid[i]] * b

            self.clip_feat[valid[i]] = clip_feat.T * a + self.clip_feat[valid[i]] * b
            self.weight.copy_(new_weight)

    def extract_mesh(self):
        tsdf = self.tsdf.masked_fill(self.weight == 0, torch.nan)
        tsdf = tsdf.cpu()

        verts, faces, _, _ = skimage.measure.marching_cubes(
            tsdf.cpu().view(*self.nvox).numpy(), level=0
        )
        good_faces = ~np.any(np.isnan(verts[faces]), axis=(1, 2))
        faces = faces[good_faces]

        verts_used_idx = np.unique(faces.flatten())
        verts_used_mask = np.zeros(len(verts), dtype=bool)
        verts_used_mask[verts_used_idx] = True

        reindex = np.cumsum(verts_used_mask) - 1
        faces = reindex[faces]
        verts = verts[verts_used_mask]

        grid = (verts + 0.5) / self.nvox * 2 - 1
        grid = grid[..., [2, 1, 0]].float()

        grid = grid.to(self.rgb.device)

        vertex_colors = torch.nn.functional.grid_sample(
            self.rgb.T.view(3, *self.nvox)[None],
            grid[None, None, None],
            align_corners=False,
            mode="bilinear",
        )[0, :, 0, 0]
        vertex_colors = vertex_colors.T.clamp(0, 1)

        vertex_clip_feats = torch.nn.functional.grid_sample(
            self.clip_feat.T.view(self.n_clip_feats, *self.nvox)[None],
            grid[None, None, None],
            align_corners=False,
            mode="bilinear",
        )[0, :, 0, 0]
        vertex_clip_feats = vertex_clip_feats.T

        verts_world = verts * self.voxel_size + self.origin.numpy()
        return verts_world, faces, vertex_colors, vertex_clip_feats


class Clip(torch.nn.Module):
    def __init__(self, clip_model, pretraining):
        super().__init__()
        self.clip = open_clip.create_model(
            clip_model, pretrained=pretraining, require_pretrained=True
        )
        self.tokenizer = open_clip.get_tokenizer(clip_model)
        self.channel_mean = torch.nn.Parameter(
            torch.tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None],
            requires_grad=False,
        )
        self.channel_std = torch.nn.Parameter(
            torch.tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None],
            requires_grad=False,
        )
        self.feature_dim = self.clip.visual.output_dim

    def normalize_img(self, rgb_img_0_1):
        return (rgb_img_0_1 - self.channel_mean) / self.channel_std

    def unnormalize_img(self, rgb_img_normed):
        return rgb_img_normed * self.channel_std + self.channel_mean

    def get_patches(self, rgb_imgs, patch_size, patch_stride):
        batch_size, _, imheight, imwidth = rgb_imgs.shape

        assert (imheight - patch_size) % patch_stride == 0
        assert (imwidth - patch_size) % patch_stride == 0

        npatches_x = int(np.round(1 + (imwidth - patch_size) / patch_stride))
        npatches_y = int(np.round(1 + (imheight - patch_size) / patch_stride))
        unfold = torch.nn.Unfold(
            kernel_size=(patch_size, patch_size),
            stride=patch_stride,
        )
        patches = unfold(rgb_imgs)
        patches = patches.transpose(1, 2).view(
            batch_size, npatches_y, npatches_x, 3, patch_size, patch_size
        )

        return patches

    def img_inference_tiled(self, rgb_imgs, patch_size, patch_stride):
        rgb_imgs = self.normalize_img(rgb_imgs)

        # currently patch_size is 256, patch_stride is 128
        # meaning two patches are overlapped by 128 pixels
        patches = self.get_patches(rgb_imgs, patch_size, patch_stride)
        batch_size, npatches_y, npatches_x, _, _, _ = patches.shape

        patches = patches.reshape(
            batch_size * npatches_y * npatches_x, 3, patch_size, patch_size
        )

        # resize patches to 224x224 for CLIP input
        patches = torch.nn.functional.interpolate(
            patches, size=(224, 224), mode="bilinear", align_corners=False
        )

        clip_feats = torch.empty(len(patches), self.feature_dim, device=rgb_imgs.device)
        max_patch_batch_size = 8
        nbatches = int(np.ceil(len(patches) / max_patch_batch_size))

        # send patches to CLIP in batches
        for i in tqdm.trange(nbatches, desc="patch batches", leave=False):
            start = i * max_patch_batch_size
            stop = min(len(patches), (i + 1) * max_patch_batch_size)
            clip_feats[start:stop] = self.clip.encode_image(patches[start:stop])

        clip_feats = clip_feats.view(
            batch_size, npatches_y, npatches_x, self.feature_dim
        ).permute(0, 3, 1, 2)

        return clip_feats

    def img_inference_tiled_depthscaled(self, rgb_imgs, depth_imgs, K, patch_stride):
        rgb_imgs = self.normalize_img(rgb_imgs)
        batch_size = len(rgb_imgs)
        device = rgb_imgs.device

        patch_footprint_m = 0.5
        ycs = np.arange(patch_stride, rgb_imgs.shape[2], patch_stride)
        xcs = np.arange(patch_stride, rgb_imgs.shape[3], patch_stride)

        clip_feat_img = torch.zeros(
            batch_size, self.feature_dim, *rgb_imgs.shape[-2:], device=device
        )
        weight = torch.zeros(batch_size, *rgb_imgs.shape[-2:], device=device)

        for b in range(len(rgb_imgs)):
            for i in range(len(ycs)):
                for j in range(len(xcs)):
                    yc = ycs[i]
                    xc = xcs[j]

                    depth = depth_imgs[b, yc, xc]
                    if depth > 0:
                        fx = K[b, 0, 0]
                        fy = K[b, 1, 1]
                        patch_width = int((fx * patch_footprint_m / depth).round())
                        patch_height = int((fy * patch_footprint_m / depth).round())

                        ymin = max(0, yc - patch_height // 2)
                        ymax = yc + patch_height // 2

                        xmin = max(0, xc - patch_width // 2)
                        xmax = xc + patch_width // 2

                        patch_rgb = rgb_imgs[b, :, ymin:ymax, xmin:xmax]

                        patch_rgb = torch.nn.functional.interpolate(
                            patch_rgb[None],
                            size=(224, 224),
                            mode="bilinear",
                            align_corners=False,
                        )

                        patch_clip = self.clip.encode_image(patch_rgb)
                        clip_feat_img[b, :, ymin:ymax, xmin:xmax] += patch_clip[
                            0, :, None, None
                        ]
                        weight[b, ymin:ymax, xmin:xmax] += 1

        clip_feat_img /= weight + (weight == 0)
        return clip_feat_img

    def text_inference(self, str_list):
        device = list(self.clip.parameters())[0].device
        tokens = self.tokenizer(str_list).to(device)
        clip_text_features = self.clip.encode_text(tokens)
        clip_text_features /= clip_text_features.norm(dim=-1, keepdim=True)
        return clip_text_features

    def run_query(self, img_feats, labels):
        clip_feat_dim = img_feats.shape[-1]
        clip_text_features = self.text_inference(labels)[:, :clip_feat_dim]
        dotprod = 100 * (img_feats @ clip_text_features.T)
        relevance = dotprod.softmax(dim=-1)
        return relevance

    @staticmethod
    def clip_feature_surgery(image_features, text_features, redundant_feats=None, t=2):
        if redundant_feats != None:
            similarity = image_features @ (text_features - redundant_feats).t()

        else:
            # weights to restrain influence of obvious classes on others
            prob = image_features[:, :1, :] @ text_features.t()
            prob = (prob * 2).softmax(-1)
            w = prob / prob.mean(-1, keepdim=True)

            # element-wise multiplied features
            b, n_t, n_i, c = (
                image_features.shape[0],
                text_features.shape[0],
                image_features.shape[1],
                image_features.shape[2],
            )
            feats = image_features.reshape(b, n_i, 1, c) * text_features.reshape(
                1, 1, n_t, c
            )
            feats *= w.reshape(1, 1, n_t, 1)
            redundant_feats = feats.mean(2, keepdim=True)  # along cls dim
            feats = feats - redundant_feats

            # sum the element-wise multiplied features as cosine similarity
            similarity = feats.sum(-1)

        return similarity

    def encode_text_with_prompt_ensemble(self, texts, device, prompt_templates=None):
        # using default prompt templates for ImageNet
        if prompt_templates == None:
            prompt_templates = [
                "a bad photo of a {}.",
                "a photo of many {}.",
                "a sculpture of a {}.",
                "a photo of the hard to see {}.",
                "a low resolution photo of the {}.",
                "a rendering of a {}.",
                "graffiti of a {}.",
                "a bad photo of the {}.",
                "a cropped photo of the {}.",
                "a tattoo of a {}.",
                "the embroidered {}.",
                "a photo of a hard to see {}.",
                "a bright photo of a {}.",
                "a photo of a clean {}.",
                "a photo of a dirty {}.",
                "a dark photo of the {}.",
                "a drawing of a {}.",
                "a photo of my {}.",
                "the plastic {}.",
                "a photo of the cool {}.",
                "a close-up photo of a {}.",
                "a black and white photo of the {}.",
                "a painting of the {}.",
                "a painting of a {}.",
                "a pixelated photo of the {}.",
                "a sculpture of the {}.",
                "a bright photo of the {}.",
                "a cropped photo of a {}.",
                "a plastic {}.",
                "a photo of the dirty {}.",
                "a jpeg corrupted photo of a {}.",
                "a blurry photo of the {}.",
                "a photo of the {}.",
                "a good photo of the {}.",
                "a rendering of the {}.",
                "a {} in a video game.",
                "a photo of one {}.",
                "a doodle of a {}.",
                "a close-up photo of the {}.",
                "a photo of a {}.",
                "the origami {}.",
                "the {} in a video game.",
                "a sketch of a {}.",
                "a doodle of the {}.",
                "a origami {}.",
                "a low resolution photo of a {}.",
                "the toy {}.",
                "a rendition of the {}.",
                "a photo of the clean {}.",
                "a photo of a large {}.",
                "a rendition of a {}.",
                "a photo of a nice {}.",
                "a photo of a weird {}.",
                "a blurry photo of a {}.",
                "a cartoon {}.",
                "art of a {}.",
                "a sketch of the {}.",
                "a embroidered {}.",
                "a pixelated photo of a {}.",
                "itap of the {}.",
                "a jpeg corrupted photo of the {}.",
                "a good photo of a {}.",
                "a plushie {}.",
                "a photo of the nice {}.",
                "a photo of the small {}.",
                "a photo of the weird {}.",
                "the cartoon {}.",
                "art of the {}.",
                "a drawing of the {}.",
                "a photo of the large {}.",
                "a black and white photo of a {}.",
                "the plushie {}.",
                "a dark photo of a {}.",
                "itap of a {}.",
                "graffiti of the {}.",
                "a toy {}.",
                "itap of my {}.",
                "a photo of a cool {}.",
                "a photo of a small {}.",
                "a tattoo of the {}.",
                "there is a {} in the scene.",
                "there is the {} in the scene.",
                "this is a {} in the scene.",
                "this is the {} in the scene.",
                "this is one {} in the scene.",
            ]

        text_features = []
        model_device = list(self.clip.parameters())[0].device
        for t in texts:
            prompted_t = [template.format(t) for template in prompt_templates]
            prompted_t = self.tokenizer(prompted_t)  # .to(device)
            class_embeddings = self.clip.encode_text(prompted_t.to(model_device))
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            text_features.append(class_embedding)
        text_features = torch.stack(text_features, dim=1).to(device).t()

        return text_features


def run_clipfusion(scan_dir, output_dir, config, device):
    if config["dataset"] == "iphone":
        dataset = iPhone3DScannerDataset(scan_dir, 0)
    elif config["dataset"] == "scannet":
        dataset = ScanNetDataset(scan_dir)
    elif config["dataset"] == "hypersim":
        dataset = HypersimDataset(scan_dir)
    elif config["dataset"] == "lerf":
        dataset = LERFDataset(scan_dir)
    else:
        raise NotImplementedError

    scan_name = os.path.basename(scan_dir)

    scene_outputdir = os.path.join(
        output_dir,
        scan_name,
    )
    os.makedirs(scene_outputdir, exist_ok=True)

    # get scene bounds
    xyz, rgb = backproject_pcd(
        dataset, batch_size=1, num_workers=4, device="cpu", max_depth=4
    )

    # here exports a sparse point cloud of the scene for preview
    if False:
        # npts = min(len(xyz), 2**16)
        # idx = np.random.choice(len(xyz), size=npts, replace=False)
        idx = np.arange(len(xyz))

        pcd = trimesh.PointCloud(xyz[idx].numpy(), colors=rgb[idx].numpy())
        _ = pcd.export("pcd.ply")

    """
    imheight = 768
    imwidth = 1024
    cfgs = []
    for patch_size in range(64, imheight):
        for patch_stride in range(patch_size // 4, patch_size):
            a = (imwidth - patch_size) % patch_stride == 0
            b = (imheight - patch_size) % patch_stride == 0
            if a and b:
                npatches_x = (imwidth - patch_size) // patch_stride + 1
                npatches_y = (imheight - patch_size) // patch_stride + 1
                print(patch_size, patch_stride, npatches_x, npatches_y)
                cfgs.append((patch_size, patch_stride))

    for clip_patch_size, clip_patch_stride in cfgs:
    for clip_model, clip_pretraining in [tup for tup in open_clip.list_pretrained() if 'ViT-B-16' in tup[0]]:
    """

    with open(os.path.join(scene_outputdir, "config.yml"), "w") as f:
        yaml.dump(config, f)

    scale_patches_by_depth = False
    trunc_m = config["trunc_vox"] * config["voxel_size"]

    minbound = torch.tensor(np.percentile(xyz.cpu(), 1, axis=0)).float() - trunc_m
    maxbound = torch.tensor(np.percentile(xyz.cpu(), 99, axis=0)).float() + trunc_m

    # number of voxels is determined by the scene bounds and voxel size
    # nvox is tensor([57, 50, 34], dtype=torch.int32)
    # nvox is the number of voxels in each dimension
    nvox = ((maxbound - minbound) / config["voxel_size"]).round().int()

    clip_fusion = ClipFusion(
        minbound,
        config["voxel_size"],
        nvox,
        trunc_m,
        scale_patches_by_depth,
        config["clip_model"],
        config["clip_pretraining"],
        config["clip_patch_size"],
        config["clip_patch_stride"],
    )
    clip_fusion = clip_fusion.to(device)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4)

    write_incremental_meshes = False
    # write_incremental_meshes = True

    for view in tqdm.tqdm(loader, leave=False, desc="run fusion"):
        rgb_imgs, depth_imgs, poses, K, frame_idx = view

        depth_imgs = depth_imgs.to(device)
        rgb_imgs = rgb_imgs.to(device)
        poses = poses.to(device)
        K = K.to(device)

        clip_fusion.integrate(depth_imgs, rgb_imgs, poses, K)

        # incrementally visualize the mesh with more views coming in
        if write_incremental_meshes:
            verts, faces, vertex_colors, vertex_clip_feats = clip_fusion.extract_mesh()

            use_rgb = True
            if use_rgb:
                vertex_colors = vertex_colors.cpu()
            else:
                query = "a musical instrument"
                labels = [
                    "an object",
                    "objects",
                    "a thing",
                    "things",
                    "stuff",
                    "texture",
                    query,
                ]
                labels = [f"a picture of {label}" for label in labels]
                vertex_clip_feats /= vertex_clip_feats.norm(dim=-1, keepdim=True)
                vertex_clip_feats = vertex_clip_feats.cpu()
                relevance = clip_fusion.clip.run_query(
                    vertex_clip_feats.to(device), labels
                )[:, -1].cpu()
                vertex_colors = plt.cm.turbo(relevance)[:, :3]

            mesh = trimesh.Trimesh(
                vertices=verts,
                faces=faces,
                vertex_colors=vertex_colors,
            )
            _ = mesh.export(f"meshes/{str(frame_idx[-1].item()).zfill(4)}.ply")

    verts, faces, vertex_colors, vertex_clip_feats = clip_fusion.extract_mesh()
    vertex_colors = vertex_colors.cpu().numpy()
    vertex_clip_feats = vertex_clip_feats.cpu().numpy()

    mesh_rgb = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        vertex_colors=vertex_colors,
    )
    _ = mesh_rgb.export(os.path.join(scene_outputdir, "mesh_rgb.ply"))

    np.save(os.path.join(scene_outputdir, "vertex_clip_feats.npy"), vertex_clip_feats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scan_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--notes")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--clip-model", default="ViT-B-32-quickgelu")
    parser.add_argument("--clip-pretraining", default="laion400m_e32")
    parser.add_argument("--voxel-size", type=float, default=0.04)
    parser.add_argument("--scan-name", help="restrict to a single scan")
    parser.add_argument("--clip-patch-size", type=int)
    parser.add_argument("--clip-patch-stride", type=int)
    args = parser.parse_args()

    if "scannet" in args.scan_dir:
        dataset_name = "scannet"
        if args.clip_patch_size is None:
            args.clip_patch_size = 160
        if args.clip_patch_stride is None:
            args.clip_patch_stride = 80
        trunc_vox = 3
    elif "hypersim" in args.scan_dir:
        dataset_name = "hypersim"
        if args.clip_patch_size is None:
            args.clip_patch_size = 256
        if args.clip_patch_stride is None:
            args.clip_patch_stride = 128
        trunc_vox = 2
    elif "lerf" in args.scan_dir:
        dataset_name = "lerf"
        if args.clip_patch_size is None:
            args.clip_patch_size = 64
        if args.clip_patch_stride is None:
            args.clip_patch_stride = 32
        trunc_vox = 3
    else:
        raise Exception("could not identify dataset from path")

    scan_dirs = sorted(
        [d for d in glob.glob(os.path.join(args.scan_dir, "*")) if os.path.isdir(d)]
    )

    print(f"dataset: {dataset_name}")
    print(f"found {len(scan_dirs)} scans")

    if args.scan_name is not None:
        for scan_dir in scan_dirs:
            if os.path.basename(scan_dir) == args.scan_name:
                scan_dirs = [scan_dir]
                print(f"limiting to scan: {args.scan_name}")
                break
        else:
            raise Exception(f"couldn't find a scan called {args.scan_name}")

    config = {
        "clip_model": args.clip_model,
        "clip_pretraining": args.clip_pretraining,
        "clip_patch_size": args.clip_patch_size,
        "clip_patch_stride": args.clip_patch_stride,
        "trunc_vox": trunc_vox,
        "voxel_size": args.voxel_size,
        "dataset": dataset_name,
    }
    if args.notes is not None:
        config["notes"] = args.notes
    print("CLIP config:", config)

    for scan_dir in tqdm.tqdm(scan_dirs):
        run_clipfusion(scan_dir, args.output_dir, config, args.device)


if __name__ == "__main__":
    main()
