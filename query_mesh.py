import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import trimesh
import clipfusion
from clipfusion import control_objects

target_objects = ["floor", "computer screen", "keyboard", "rug", "sharp corners"]
# target_objects = ["computer screen"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_dir")
    # parser.add_argument("query")
    # parser.add_argument("--outfile", "-o")
    args = parser.parse_args()

    vertex_clip_feats = torch.from_numpy(
        np.load(os.path.join(args.pred_dir, "vertex_clip_feats.npy"))
    )
    feat_norm = vertex_clip_feats.norm(dim=-1, keepdim=True)
    vertex_clip_feats /= feat_norm
    clip_feat_dim = vertex_clip_feats.shape[1]

    config_file = os.path.join(args.pred_dir, "config.yml")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    clip = clipfusion.Clip(config["clip_model"], config["clip_pretraining"])
    clip.requires_grad_(False)
    clip.eval()

    for object in target_objects:
        labels = ["an object", "things", "stuff", "texture", object]
        labels = [f"a picture of {label}" for label in labels]
        relevance = clip.run_query(vertex_clip_feats, labels)[:, -1]
        relevance = ((relevance - 0.5) * 2).clamp(0, 1)

        # read the ply file with vedo
        mesh = trimesh.load_mesh(os.path.join(args.pred_dir, "mesh_rgb.ply"))
        mesh.visual.vertex_colors = plt.cm.turbo(relevance)[:, :3] * 255

        outfile = os.path.join(args.pred_dir, "clipfusion_" + object + ".ply")
        print(f"saving to {outfile}")
        mesh.export(outfile)

    ### Clip Surgery method

    # Prompt ensemble for text features with normalization
    text_features = clip.encode_text_with_prompt_ensemble(
        control_objects, "cpu", prompt_templates=["there is a {} in the scene."]
    )

    similarity = clip.clip_feature_surgery(vertex_clip_feats[None], text_features)

    # Normalize similarity to [0, 1]
    similarity = (similarity - similarity.min(1, keepdim=True)[0]) / (
        similarity.max(1, keepdim=True)[0] - similarity.min(1, keepdim=True)[0]
    )

    for n in range(similarity.shape[-1]):
        if control_objects[n] not in target_objects:
            continue

        relevance = similarity[0, :, n]
        median = torch.median(relevance)
        std = torch.std(relevance)
        # values under 2 sigma above the mean are set to 0
        relevance_clipped = torch.where(
            relevance > median + 2 * std, relevance, torch.zeros_like(relevance)
        )

        # read the ply file with vedo
        mesh = trimesh.load_mesh(os.path.join(args.pred_dir, "mesh_rgb.ply"))
        mesh.visual.vertex_colors = plt.cm.turbo(relevance_clipped)[:, :3] * 255

        outfile = os.path.join(
            args.pred_dir, "clipSurgery_" + control_objects[n] + ".ply"
        )
        print(f"saving to {outfile}")
        mesh.export(outfile)
