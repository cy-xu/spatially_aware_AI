import sys
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import skimage.measure
import torch
import tqdm
import yaml
from vedo import *
import trimesh
import open3d as o3d
import pretty_errors

from clipfusion import (
    LERFDataset,
    ScanNetDataset,
    HypersimDataset,
    backproject_pcd,
    Clip,
    iPhone3DScannerDataset,
    control_objects,
)

from dgcnn.main_cls import InSituLearning


from handy_utils import (
    dotdict,
    get_path,
    flood_fill_3d,
    extract_mesh_by_object,
    KmaxSegmentationModel,
    mesh_to_json,
)


class InSituManager:
    """
    a new class that wraps clip_fuison and in-situ learning
    mainly a unified interface for the web app to communicate with
    Unity and user devices
    """

    def __init__(
        self,
        scan_dir="scenes/iphone_3dscanner",
        scan_name="5110_my_corner_v0",
        dataset="iphone",
        output_dir="unity_output",
        device="cuda",
        clip_model="ViT-B-32-quickgelu",
        clip_pretraining="laion400m_e32",
        voxel_size=0.04,
        trunc_vox=2,
        clip_patch_size=128,
        clip_patch_stride=64,
        curr_ver=0,
    ):
        # collect all config in a dict for cleaner code
        config = {
            # scene config
            "scan_dir": scan_dir,
            "scan_name": scan_name,
            "device": device,
            "output_dir": output_dir,
            # clip fusion config
            "clip_model": clip_model,
            "clip_pretraining": clip_pretraining,
            "clip_patch_size": clip_patch_size,
            "clip_patch_stride": clip_patch_stride,
            "trunc_vox": trunc_vox,
            "voxel_size": voxel_size,
            "dataset": dataset,
            # seem config
            "seg_conf_file": "kmax/kmax_convnext_large.yaml",
            "seg_model_path": "kmax/kmax_convnext_large.pth",
            # in-situ learning config
            "instu_model": "dgcnn",
            "use_sgd": False,
            "lr": 0.001,
            "scheduler": "cosine",
            "batch_size": 128,
            "epochs": 200,
            "momentum": 0.9,
            "dropout": 0.5,
            "emb_dims": 1024,
            "seed": 1,
            "num_points": 30,
            "k_neighbors": 20,
        }
        self.config = dotdict(config)
        self.curr_ver = curr_ver

        self.scene_knowledge = None
        self.scene_knowledge_prev = None
        self.scene_reconstructed = False

        # update configs and check if the scene if already processed
        self.update_config(target_version=curr_ver)
        config = self.config

        # version control of the scene
        # iphone_3dscanner/5110_my_corner/v00 --> unity_output/5110_my_corner/v00
        self.scan_versions = sorted(
            [
                os.path.basename(v)
                for v in glob.glob(os.path.join(scan_dir, scan_name, "v*"))
            ]
        )

        # clip model always inits
        self.clip_model = Clip(config.clip_model, config.clip_pretraining)
        self.clip_model.requires_grad_(False)
        self.clip_model.eval()

        # Prompt ensemble for text features with normalization
        # self.control_objects = control_objects
        # self.control_objects = ["thing", "object", "stuff"]
        self.control_objects = None
        self.control_text_features = None

        if self.control_objects:
            self.control_text_features = (
                self.clip_model.encode_text_with_prompt_ensemble(
                    self.control_objects,
                    "cpu",
                    prompt_templates=["there is a {} in the scene."],
                )
            )

        self.seg_model = KmaxSegmentationModel(
            config.seg_conf_file, config.seg_model_path, config.device
        )
        self.stuff_classes = self.seg_model.metadata.stuff_classes

        # self.seg_model = self.seem_init(config.seem_conf_file, config.seem_model_path)

        # init and load the in-situ model
        self.insitu_model = InSituLearning(
            model=config.instu_model,
            device=config.device,
            emb_dims=config.emb_dims,
            k_neighbors=config.k_neighbors,
            dropout=config.dropout,
            output_channels=50,
            num_points=config.num_points,
            cool_down_epochs=10,
            batch_size=config.batch_size,
            use_sgd=config.use_sgd,
            lr=config.lr,
            momentum=config.momentum,
            epochs=config.epochs,
            model_path=config.insitu_model_path,
            label_path=config.insitu_labels,
        )

        # auto run clipfusion reconstruction if the scene is not processed
        if not self.scene_reconstructed:
            self.run_clipfusion(
                scan_dir=config.scene_inputdir,
                config=config,
                device=device,
                views_limit=0,
                scale_patches_by_depth=False,
                curr_ver=curr_ver,
            )

    def update_config(self, target_version):
        # check if version changed
        switch_version = target_version != self.curr_ver

        if switch_version:
            self.scene_knowledge_prev = self.scene_knowledge.copy()
            print("switching scene version to", target_version)

        # set all paths
        config = self.config
        config = get_path(config, target_version)

        # update version
        self.curr_ver = config.curr_ver = target_version

        # create necessary directories
        os.makedirs(
            config.scene_dir,
            exist_ok=True,
        )
        os.makedirs(config.scene_outputdir, exist_ok=True)

        # save and print the configs (create the directory first!)
        with open(os.path.join(config.scene_outputdir, "config.yml"), "w") as f:
            yaml.dump(dict(config), f)

        self.config = config
        for key, value in config.items():
            print(f"{key:<30} {value}")

        # if the scene is already processed, load the scene knowledge
        sk_path = get_path(config, target_version, "scene_knowledge")

        if os.path.exists(sk_path):
            self.scene_reconstructed = True

            print(f"\nloading scene knowledge from {sk_path}")
            with open(sk_path) as f:
                self.scene_knowledge = json.load(f)

            # voxel stuff
            self.voxel_rgb = np.load(get_path(config, target_version, "voxel_rgb"))
            self.nvox = self.voxel_rgb.shape[:3]
            self.voxel_clip_feats = np.load(
                get_path(config, target_version, "voxel_clip_feats")
            )

            # load the clip features to gpu
            self.vert_clip_feat = np.load(
                get_path(config, target_version, "vertex_clip_feats")
            )

            # mesh stuff
            self.mesh_rgb = trimesh.load(get_path(config, target_version, "mesh_rgb"))
            self.verts = self.mesh_rgb.vertices.tolist()
            self.faces = self.mesh_rgb.faces.tolist()
            self.vertex_colors = (
                np.array(self.mesh_rgb.visual.vertex_colors)[:, :3] / 255.0
            )
            self.vertex_colors = self.vertex_colors.astype(np.float32).tolist()

            # laod segmentation mesh
            self.mesh_segmentation = trimesh.load(
                get_path(config, target_version, "mesh_segmentation")
            )
            self.vertex_seg_color = (
                np.array(self.mesh_segmentation.visual.vertex_colors)[:, :3] / 255.0
            )
            self.vertex_seg_color = self.vertex_seg_color.astype(np.float32).tolist()

            self.vertex_obj_idx = np.load(
                get_path(config, target_version, "vertex_obj_idx")
            ).astype(np.int32)
        else:
            print("scene not processed yet, None")

    def run_clipfusion(
        self,
        scan_dir,
        config,
        device,
        views_limit=0,
        scale_patches_by_depth=False,
        curr_ver=0,
    ):
        scene_inputdir = get_path(self.config, curr_ver, "scene_inputdir")

        if config["dataset"] == "iphone":
            dataset = iPhone3DScannerDataset(scene_inputdir, views_limit)
        elif config["dataset"] == "magicleap2":
            dataset = iPhone3DScannerDataset(scan_dir)
        elif config["dataset"] == "scannet":
            dataset = ScanNetDataset(scan_dir)
        else:
            raise NotImplementedError

        # get scene bounds
        max_depth = 4
        xyz, rgb = backproject_pcd(
            dataset, batch_size=1, num_workers=4, device="cpu", max_depth=max_depth
        )

        # here exports a sparse point cloud of the scene for preview
        idx = np.arange(len(xyz))
        pcd = trimesh.PointCloud(xyz[idx].numpy(), colors=rgb[idx].numpy())
        _ = pcd.export("point_cloud_preview.ply")

        trunc_m = config["trunc_vox"] * config["voxel_size"]

        minbound = torch.tensor(np.percentile(xyz.cpu(), 1, axis=0)).float() - trunc_m
        maxbound = torch.tensor(np.percentile(xyz.cpu(), 99, axis=0)).float() + trunc_m
        origin = minbound

        # number of voxels is determined by the scene bounds and voxel size
        # nvox is tensor([57, 50, 34], dtype=torch.int32)
        # nvox is the number of voxels in each dimension
        nvox = ((maxbound - minbound) / config["voxel_size"]).round().int()
        self.nvox = nvox
        print(f"voxel grid shape: {nvox}")

        clip_fusion = ClipSeemFusion(
            origin,
            config["voxel_size"],
            nvox,
            trunc_m,
            scale_patches_by_depth,
            config["clip_patch_size"],
            config["clip_patch_stride"],
            self.clip_model,
            self.seg_model,
        )
        clip_fusion = clip_fusion.to(device)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1)

        for view in tqdm.tqdm(loader, leave=False, desc="run fusion"):
            rgb_imgs, depth_imgs, poses, K, frame_idx = view

            depth_imgs = depth_imgs.to(device)
            rgb_imgs = rgb_imgs.to(device)
            poses = poses.to(device)
            K = K.to(device)

            clip_fusion.integrate(depth_imgs, rgb_imgs, poses, K)

        def argmax_with_check_2d_efficient(tensor):
            any_nonzero = tensor.any(dim=1)
            max_indices = torch.argmax(tensor, dim=1)
            max_indices *= any_nonzero
            max_indices -= (~any_nonzero).long()
            return max_indices

        # now we have identified all objects in the scene, convert one-hot labels back to class ID
        self.onehot_to_index = argmax_with_check_2d_efficient(
            clip_fusion.labels_one_hot
        )

        # this line creates the bug that confuses the all zeros vector's max with the first element being the largest
        # self.onehot_to_index = torch.argmax(clip_fusion.labels_one_hot, dim=1)

        # traverse the voxel grid and build a dict of unique objects in the scene
        self.onehot_to_index = (
            self.onehot_to_index.view(*clip_fusion.nvox).cpu().numpy()
        )

        self.voxel_clip_feats = (
            clip_fusion.clip_feat.view(*clip_fusion.nvox, -1).cpu().numpy()
        )
        self.voxel_rgb = clip_fusion.rgb.view(*clip_fusion.nvox, -1).cpu().numpy()

        # building the public knowledge of the scene
        scene_knowledge, voxel_obj_idx = flood_fill_3d(
            self.onehot_to_index,
            self.scene_knowledge,
            self.voxel_clip_feats,
            self.voxel_rgb,
            self.insitu_model,
            self.scene_knowledge_prev,
        )

        # add version to scene_knowledge
        scene_knowledge["scan_version"] = curr_ver
        clip_fusion.unique_objects = scene_knowledge["unique_objects"]
        clip_fusion.voxel_obj_idx = torch.from_numpy(voxel_obj_idx).to(
            device
        )  # for mesh extraction

        # share with manager
        self.clip_fuison = clip_fusion
        self.scene_knowledge = scene_knowledge

        # colorize each unique object with a unique color in another grid
        objects_segmentation_color = torch.clone(clip_fusion.rgb).view(
            *clip_fusion.nvox, -1
        )

        # new approach, category color already saved in scene_knowledge
        for obj_key, obj_info in scene_knowledge["unique_objects"].items():
            vox_indices = tuple(zip(*obj_info["voxels"]))
            color_tensor = torch.tensor(obj_info["color"]).float() / 255.0
            objects_segmentation_color[vox_indices] = color_tensor.to(device)

        clip_fusion.objects_segmentation_color = objects_segmentation_color.view(-1, 3)

        # perform ML on the voxel grid, 20x less voxels than vertices
        # export voxel CLIP features to numpy array

        # full mesh with all views
        (
            verts,
            faces,
            vertex_colors,
            vertex_clip_feats,
            vertex_obj_idx,
            segmentation_color,
        ) = clip_fusion.extract_mesh()

        vertex_colors = vertex_colors.cpu().numpy()
        vertex_obj_idx = vertex_obj_idx.cpu().numpy()

        # seems to be correct until now
        # but no matching in extract_mesh_by_object

        # extra step to extract object meshes and store them in the scene knowledge
        for obj_key, obj_value in scene_knowledge["unique_objects"].items():
            obj_idx = obj_value["object_index"]

            obj_verts, obj_faces, obj_colors, obj_mesh = extract_mesh_by_object(
                verts,
                faces,
                vertex_colors,
                vertex_obj_idx,
                obj_idx,
            )

            if len(obj_faces) < 10:
                self.scene_knowledge["unique_objects"][obj_key]["mesh"] = None
                # print(f"{obj_key} has less than 10 faces, skipping")
                continue

            # deliver in json
            obj_mesh = {
                "vertices": obj_verts.tolist(),
                "faces": obj_faces.tolist(),
                "colors": obj_colors.tolist(),
            }

            self.scene_knowledge["unique_objects"][obj_key]["mesh"] = obj_mesh

        self.verts, self.faces = verts.tolist(), faces.tolist()
        self.vertex_colors = vertex_colors.tolist()

        self.segmentation_color = segmentation_color.cpu().numpy()
        self.vert_clip_feat = vertex_clip_feats.cpu().numpy()
        self.vertex_obj_idx = vertex_obj_idx

        self.save_files_and_broadcast(new_scene=True)

        # Get the peak GPU memory allocated
        current_max_memory = torch.cuda.max_memory_allocated()
        print(f"Max memory allocated: {current_max_memory} bytes")

        # end of clipfusion, clean GPU memory
        del clip_fusion
        # self.clip_fusion = clip_fusion
        torch.cuda.empty_cache()

        self.scene_reconstructed = True

    def request_mesh(self, version, obj_key="scene", mesh_type="rgb"):
        mesh_name = "mesh_" + mesh_type

        #  return the full scene mesh
        if obj_key == "scene":
            mesh_json = mesh_to_json(self.config, version, mesh_name)
            return mesh_json

        else:
            # extract the mesh for indivual objects

            # load correct scene knowledge version
            sk_temp = json.load(
                open(get_path(self.config, version, "scene_knowledge"), "r")
            )

            if obj_key == "all_objects":
                obj_dict = sk_temp["unique_objects"]

            elif obj_key == "unchanged":
                # sk_temp is always v01 in thsi case
                obj_dict = sk_temp["unchanged_objects"]

            elif obj_key == "missing":
                # missing objects are saved in current knowledge
                # but their mesh is extracted from the previous version
                sk_v0 = json.load(
                    open(get_path(self.config, 0, "scene_knowledge"), "r")
                )
                obj_keys = sk_temp["missing_objects"]
                obj_dict = {k: sk_v0["unique_objects"][k] for k in obj_keys}

            else:
                print(f"invalid object key: {obj_key}")

            obj_meshes = {}
            for obj_key, obj_info in obj_dict.items():
                if obj_info["mesh"] is None:
                    continue
                obj_meshes[obj_key] = obj_info["mesh"]

        return obj_meshes

    def clip_text_query(self, text: str) -> list:
        # labels = ["an object", "things", "stuff", "texture", text]

        if self.control_objects is None:
            # use very class label appearred in the scene
            uo = self.scene_knowledge["unique_objects"]
            self.control_objects = [uo[k]["class_label"] for k in uo.keys()]
            # remove duplicates
            self.control_objects = list(set(self.control_objects))

            print(
                f"unique objects found: {len(uo)}, types of objects for query control {len(self.control_objects)} :", self.control_objects
            )

        if text not in self.control_objects or self.control_text_features is None:
            self.control_objects.append(text)

            self.control_text_features = (
                self.clip_model.encode_text_with_prompt_ensemble(
                    self.control_objects,
                    "cpu",
                    prompt_templates=["a photo of {}"],
                )
            )

        clip_feat = torch.from_numpy(self.vert_clip_feat)
        feat_norm = clip_feat.norm(dim=-1, keepdim=True)
        clip_feat /= feat_norm
        # replace all nans with 0
        clip_feat = torch.nan_to_num(clip_feat)

        similarity = self.clip_model.clip_feature_surgery(
            clip_feat[None], self.control_text_features
        )

        for n in range(len(self.control_objects)):
            if self.control_objects[n] != text:
                continue

            # ClipFusion's method
            # labels = ["an object", "things", "stuff", "texture", text]
            # labels = [f"a picture of {label}" for label in labels]
            # relevance = self.clip_model.run_query(clip_feat, labels)[:, -1]
            # relevance = ((relevance - 0.5) * 2).clamp(0, 1)

            relevance = similarity[0, :, n].cpu().numpy()
            # # subtract the mean and clip to 0-1
            relevance -= relevance.mean()
            relevance = np.clip(relevance, 0, 1)
            relevance = (relevance - relevance.min()) / (
                relevance.max() - relevance.min()
            )

            ## color plan 1
            # alpha blend the vertex's original color with the relevance to make the high relevance area more red
            # red_area = np.zeros_like(self.vertex_colors)
            # red_area[:, 0] = relevance.cpu().numpy() * 255
            # blended_colors = self.vertex_colors + red_area * 0.3
            # blended_colors = np.clip(blended_colors, 0, 255)

            ## color plan 2
            # use the turbo colormap to highlight the most relevant vertices
            relevance_colors = plt.cm.turbo(relevance)[:, :3]
            # user relevance to adjust heatmap alpha
            # suppress the overall alpha to make the mesh more transparent
            alpha = relevance * 0.5
            relevance_colors = np.hstack([relevance_colors, alpha[:, None]])

            # rgb_colors = np.array(self.vertex_colors)
            # blended_colors = rgb_colors * 0.7 + relevance_colors * 0.3

            mesh_json = {
                "vertices": self.verts,
                "faces": self.faces,
                "colors": relevance_colors.tolist(),
            }

            return mesh_json

        return None

    def save_files_and_broadcast(self, new_scene=True):
        # some data are not changed when the scene is updated from user input
        if new_scene:
            ## export numpy arrays for in-situ learning
            np.save(get_path(self.config, self.curr_ver, "voxel_rgb"), self.voxel_rgb)
            np.save(
                get_path(self.config, self.curr_ver, "voxel_clip_feats"),
                self.voxel_clip_feats,
            )
            # save vertex_clip_feats for text query
            np.save(
                get_path(self.config, self.curr_ver, "vertex_clip_feats"),
                self.vert_clip_feat,
            )
            # save voxel_obj_idx for single object mesh extraction
            np.save(
                get_path(self.config, self.curr_ver, "vertex_obj_idx"),
                self.vertex_obj_idx,
            )

            ## export mesh files for visualization
            # RGB mesh
            mesh_rgb = trimesh.Trimesh(
                vertices=self.verts,
                faces=self.faces,
                vertex_colors=self.vertex_colors,
            )
            _ = mesh_rgb.export(get_path(self.config, self.curr_ver, "mesh_rgb"))

            # Segmentation mesh
            mesh_objects_segmentation = trimesh.Trimesh(
                vertices=self.verts,
                faces=self.faces,
                vertex_colors=self.segmentation_color,
            )
            export_path = os.path.join(
                self.config.scene_outputdir, f"mesh_segmentation.ply"
            )
            _ = mesh_objects_segmentation.export(export_path)

        with open(get_path(self.config, self.curr_ver, "scene_knowledge"), "w") as f:
            json.dump(self.scene_knowledge, f, default=str)

        with open(get_path(self.config, self.curr_ver, "insitu_labels"), "w") as f:
            json.dump(self.insitu_model.labels, f, default=str)


# new class for Clip and Seem fusion, inherit from ClipFusion
class ClipSeemFusion(torch.nn.Module):
    def __init__(
        self,
        origin,
        voxel_size,
        nvox,
        trunc,
        scale_patches_by_depth,
        clip_patch_size,
        clip_patch_stride,
        clip_model,
        seg_model,
    ):
        super().__init__()

        self.clip = clip_model

        # minbound is the origin of the scene
        self.origin = origin
        self.voxel_size = voxel_size
        self.nvox = nvox
        self.trunc = trunc
        self.clip_patch_size = clip_patch_size
        self.clip_patch_stride = clip_patch_stride
        self.scale_patches_by_depth = scale_patches_by_depth
        self.segmentation_model = seg_model

        self.n_clip_feats = self.clip.feature_dim

        self.register_buffer("tsdf", torch.zeros(torch.prod(nvox), dtype=torch.float32))
        self.register_buffer(
            "rgb", torch.zeros((torch.prod(nvox), 3), dtype=torch.float32)
        )
        self.register_buffer(
            "clip_feat",
            torch.zeros((torch.prod(nvox), self.n_clip_feats), dtype=torch.float32),
        )
        self.register_buffer("weight", torch.zeros(torch.prod(nvox), dtype=torch.int32))
        self.register_buffer(
            "tsdf_weight", torch.zeros(torch.prod(nvox), dtype=torch.int32)
        )

        # 133 is the null class, this fix the bug of confusing person (0) with null (133)
        # store some extra possible class labels for each voxel
        self.n_classes = 133 + 10
        self.register_buffer(
            "labels_one_hot",
            torch.zeros((torch.prod(nvox), self.n_classes), dtype=torch.int32),
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

        self.debug_counter = 0

    def integrate(self, depth_imgs, rgb_imgs, poses, K):
        batch_size, imheight, imwidth, _ = rgb_imgs.shape
        device = rgb_imgs.device
        rgb_imgs_orig = rgb_imgs.clone()

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

        # K is the camera intrinsic matrix
        # uvz is the pixel coordinates (u, v) and depth (z) of each point in the image

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
        valid = _valid & (sdf.abs() <= 1)
        tsdf_valid = _valid & (sdf > -1)

        batch_tsdf_weight = tsdf_valid.sum(dim=0)
        batch_tsdf_valid = batch_tsdf_weight > 0
        tsdf.masked_fill_(~tsdf_valid, 0)
        batch_tsdf = tsdf.sum(dim=0)

        # TODO: what is the weight for? alpablending for updating?
        new_weight = self.tsdf_weight + batch_tsdf_weight.to(self.tsdf_weight.dtype)

        a = new_weight[batch_tsdf_valid]
        b = (self.tsdf_weight / new_weight)[batch_tsdf_valid]

        self.tsdf[batch_tsdf_valid] = (
            batch_tsdf[batch_tsdf_valid] / a + self.tsdf[batch_tsdf_valid] * b
        )
        self.tsdf_weight.copy_(new_weight)

        """
        grid specifies the sampling pixel locations normalized by the input spatial dimensions. Therefore, it should have most values in the range of [-1, 1]. For example, values x = -1, y = -1 is the left-top pixel of input, and values x = 1, y = 1 is the right-bottom pixel of input.
        """

        rgb_imgs = rgb_imgs.permute(0, 3, 1, 2)
        for i in range(batch_size):
            grid_valid = grid[i].T[valid[i]]

            # panoptic segmentation
            pano_seg = self.segmentation_model.run_on_image(rgb_imgs[i])
            # (Pdb) pano_seg.shape --> torch.Size([768, 1024])
            # already resized to original

            # pano_seg is now the class ID for each pixel
            pano_seg_mask = pano_seg.float()

            ## debug preview

            # rgb_imgs[i] is [3, 768, 1024] tensor, save for preview
            # plt.imsave(
            #     f"debug_preview/rgb_imgs_{self.debug_counter}.png",
            #     rgb_imgs[i].cpu().numpy().transpose(1, 2, 0),
            # )

            # save pano_seg_full as png for preview
            # pano_seg_preview = pano_seg_mask.cpu().numpy()
            # pano_seg_preview = pano_seg_preview / 128 * 255
            # pano_seg_preview = pano_seg_preview.astype(np.uint8)
            # # pano_seg_preview is grayscale, apply colormap
            # pano_seg_preview = plt.cm.turbo(pano_seg_preview)
            # plt.imsave(
            #     f"debug_preview/pano_seg_preview_{self.debug_counter}.png",
            #     pano_seg_preview,
            # )

            ## TODO: use instance information to separate objects

            # self.debug_counter += 1

            # fuze object labels to the tsdf voxel grid
            labels = torch.nn.functional.grid_sample(
                pano_seg_mask[None, None],
                grid_valid[None, None],
                mode="nearest",
                align_corners=False,
            )[0, :, 0]

            rgb = torch.nn.functional.grid_sample(
                rgb_imgs[i, None],
                grid_valid[None, None],
                mode="bilinear",
                align_corners=False,
            )[0, :, 0]

            clip_feat = torch.nn.functional.grid_sample(
                clip_feat_img[i, None, : self.n_clip_feats],
                grid_valid[None, None],
                mode="bilinear",
                align_corners=False,
            )[0, :, 0]

            # TODO: a, b, and new weight for updating?
            new_weight = self.weight + valid[i].to(self.weight.dtype)
            a = 1 / new_weight[valid[i], None]
            b = self.weight[valid[i], None] * a
            self.rgb[valid[i]] = rgb.T * a + self.rgb[valid[i]] * b

            self.clip_feat[valid[i]] = clip_feat.T * a + self.clip_feat[valid[i]] * b
            self.weight.copy_(new_weight)

            # instead of averaging like RGB or clip features
            # we count the label index for each voxel
            # ID conversion, from object ID to class ID

            self.labels_one_hot[valid[i]] += torch.nn.functional.one_hot(
                labels[i].to(torch.long), num_classes=self.n_classes
            )

    def extract_mesh(self):
        # place Nan in the tsdf where the weight is 0
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

        vertex_obj_idx = torch.nn.functional.grid_sample(
            self.voxel_obj_idx[None, None].float(),
            grid[None, None, None],
            align_corners=False,
            mode="nearest",
        )[0, :, 0, 0]
        vertex_obj_idx = vertex_obj_idx.T

        vertex_segment_color = torch.nn.functional.grid_sample(
            self.objects_segmentation_color.T.view(3, *self.nvox)[None],
            grid[None, None, None],
            align_corners=False,
            mode="nearest",
        )[0, :, 0, 0]
        vertex_segment_color = vertex_segment_color.T.clamp(0, 1)

        verts_world = verts * self.voxel_size + self.origin.numpy()
        return (
            verts_world,
            faces,
            vertex_colors,
            vertex_clip_feats,
            vertex_obj_idx,
            vertex_segment_color,
        )

    # def extract_object_mesh(self, obj_idx):
    #     # re-center target surface at 0
    #     tsdf = self.voxel_obj_idx - obj_idx
    #     # place Nan in the tsdf where the weight is 0
    #     mask = (self.weight == 0).view(*self.nvox)
    #     tsdf = self.voxel_obj_idx.float().masked_fill(mask, torch.nan)
    #     tsdf = tsdf.cpu().numpy()

    #     verts, faces, _, _ = skimage.measure.marching_cubes(tsdf, level=0)
    #     good_faces = ~np.any(np.isnan(verts[faces]), axis=(1, 2))
    #     faces = faces[good_faces]

    #     if len(faces) == 0:
    #         return None

    #     verts_used_idx = np.unique(faces.flatten())
    #     verts_used_mask = np.zeros(len(verts), dtype=bool)
    #     verts_used_mask[verts_used_idx] = True

    #     reindex = np.cumsum(verts_used_mask) - 1
    #     faces = reindex[faces]
    #     verts = verts[verts_used_mask]

    #     grid = (verts + 0.5) / self.nvox * 2 - 1
    #     grid = grid[..., [2, 1, 0]].float()

    #     grid = grid.to(self.rgb.device)

    #     vertex_colors = torch.nn.functional.grid_sample(
    #         self.rgb.T.view(3, *self.nvox)[None],
    #         grid[None, None, None],
    #         align_corners=False,
    #         mode="bilinear",
    #     )[0, :, 0, 0]
    #     vertex_colors = vertex_colors.T.clamp(0, 1)

    #     verts_world = verts * self.voxel_size + self.origin.numpy()
    #     return verts_world, faces, vertex_colors
