import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm
import trimesh
import yaml

import clipfusion

eval_output_dir = "/Users/noah/output/ClipFusion/hypersim_eval"
hypersim_reconstruction_dir = "/Users/noah/output/ClipFusion/output3d/hypersim"
hypersim_data_dir = "/Users/noah/data/hypersim"

labels = pd.read_csv("hypersim_labels.csv")

for row in labels.itertuples():
    imgfile = os.path.join(hypersim_data_dir, row.scene_name, 'images/scene_cam_00_final_preview', row.file_name)
    img = imageio.imread(imgfile)
    plt.figure()
    plt.imshow(img)
    plt.title(row.label)
    plt.plot(row.x, row.y, 'r.')






tp = fp = fn = 0

for scan_name in labels.scene_name.unique():
    scan_recon_dir = os.path.join(hypersim_reconstruction_dir, scan_name)

    try:
        mesh = trimesh.load_mesh(os.path.join(scan_recon_dir, "mesh_rgb.ply"))
        vertex_clip_feats = np.load(
            os.path.join(scan_recon_dir, "vertex_clip_feats.npy")
        )
        with open(os.path.join(scan_recon_dir, "config.yml"), "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"failed to load reconstruction for {scan_name}")
        raise e

    vertex_clip_feats = torch.from_numpy(vertex_clip_feats)
    feat_norm = vertex_clip_feats.norm(dim=-1, keepdim=True)
    vertex_clip_feats /= torch.clamp_min(feat_norm, 0.1)

    scan_eval_dir = os.path.join(eval_output_dir, scan_name)
    os.makedirs(scan_eval_dir, exist_ok=True)

    label_names = list(labels.label)
    label_presence = torch.tensor(list(labels.scene_name == scan_name))

    clip = clipfusion.Clip(config["clip_model"], config["clip_pretraining"])
    clip.requires_grad_(False)
    clip.eval()

    bg_clip_text_features = clip.text_inference(
        [
            "a picture of an object",
            "a picture of things",
            "a picture of stuff",
            "a picture of texture",
        ]
    )
    target_clip_text_features = clip.text_inference(
        [f"a picture of {i}" for i in label_names]
    )
    thresholds = torch.linspace(0, 1, 101)
    preds = []
    for i in tqdm.trange(len(labels), leave=False, desc=scan_name):
        clip_text_features = torch.cat(
            (bg_clip_text_features, target_clip_text_features[i, None]), dim=0
        )
        dotprod = vertex_clip_feats @ clip_text_features.T
        relevance = (100 * dotprod).softmax(dim=-1)[..., -1]

        _ = trimesh.Trimesh(
            mesh.vertices,
            mesh.faces,
            vertex_colors=plt.cm.turbo((2 * relevance - 1).clamp_min(0))[:, :3],
        ).export(os.path.join(scan_eval_dir, label_names[i].replace(" ", "_") + ".ply"))

        pred = relevance.max() > thresholds
        preds.append(pred)

    preds = torch.stack(preds)

    tp += (preds & label_presence[:, None]).sum(dim=0)
    fp += (preds & (~label_presence[:, None])).sum(dim=0)
    fn += ((~preds) & label_presence[:, None]).sum(dim=0)

prec = tp / (tp + fp)
rec = tp / (tp + fn)

