import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

import clipfusion


class Dataset(torch.utils.data.Dataset):
    def __init__(self, imgfiles):
        self.imgfiles = imgfiles

    def __len__(self):
        return len(self.imgfiles)

    def __getitem__(self, i):
        img_npy = cv2.imread(self.imgfiles[i])
        img_npy = cv2.cvtColor(img_npy, cv2.COLOR_BGR2RGB)
        img_npy = cv2.resize(img_npy, (640, 480), None, 0, 0, cv2.INTER_AREA)

        img = torch.from_numpy(img_npy).float().permute(2, 0, 1) / 255
        return img


if __name__ == "__main__":
    device = "cpu"

    patch_size = 160
    patch_stride = 80

    clip_model = "ViT-B-32-quickgelu"
    clip_pretraining = "laion400m_e32"

    clip = clipfusion.Clip(clip_model, clip_pretraining)
    clip.requires_grad_(False)
    clip.eval()
    clip = clip.to(device)

    labels = ["object", "things", "stuff", "texture", "bed", "guitar"]
    tokens = clip.tokenizer(labels).to(device)
    clip_text_features = clip.clip.encode_text(tokens)
    clip_text_features /= clip_text_features.norm(dim=-1, keepdim=True)

    imgfiles = sorted(
        glob.glob("/Users/noah/data/scannet/scans/scene0000_00/color/*.jpg"),
        key=lambda f: int(os.path.basename(f).split(".")[0]),
    )[800:1250]
    dset = Dataset(imgfiles)
    loader = torch.utils.data.DataLoader(dset, batch_size=1, num_workers=0)

    outdir = "output2d"
    os.makedirs(outdir, exist_ok=True)
    for f in ["fig0", "fig1", "fig2"]:
        os.makedirs(os.path.join(outdir, f), exist_ok=True)

    for img_idx, img in enumerate(tqdm.tqdm(loader)):
        img = img.to(device)

        patches = clip.get_patches(img, patch_size, patch_stride)
        npatches_y = patches.shape[1]
        npatches_x = patches.shape[2]
        npatches = npatches_x * npatches_y

        clip_feats = clip.img_inference_tiled(img, patch_size, patch_stride)
        clip_feats /= torch.norm(clip_feats, dim=1, keepdim=True)

        dotprod = clip_feats.permute(0, 2, 3, 1) @ clip_text_features.T
        relevance = (100 * dotprod).softmax(dim=-1)

        rel_map = torch.nn.functional.interpolate(
            relevance.permute(0, 3, 1, 2),
            size=img.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        relevance = relevance.cpu()
        rel_map = rel_map.cpu()
        patches = patches.cpu()

        plt.figure(1)
        k = 0
        for i in range(npatches_y):
            for j in range(npatches_x):
                k += 1
                plt.subplot(npatches_y, npatches_x, k)
                plt.imshow(patches[0, i, j].permute(1, 2, 0))
                plt.axis("off")
        plt.savefig(os.path.join(outdir, "fig0", f"{str(img_idx).zfill(4)}.jpg"))
        plt.close()

        plt.figure(2)
        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.imshow(relevance[0, ..., i], vmin=0, vmax=1)
            plt.axis("off")
            plt.title(labels[i])
        plt.savefig(os.path.join(outdir, "fig1", f"{str(img_idx).zfill(4)}.jpg"))
        plt.close()

        plt.figure(3)
        plt.imshow(img.cpu()[0].permute(1, 2, 0))
        plt.imshow(rel_map[0, -1], alpha=0.5, vmin=0, vmax=1)
        plt.savefig(os.path.join(outdir, "fig2", f"{str(img_idx).zfill(4)}.jpg"))
        plt.close()

    f0 = sorted(glob.glob(os.path.join(outdir, "fig0/*.jpg")))
    f1 = sorted(glob.glob(os.path.join(outdir, "fig1/*.jpg")))
    f2 = sorted(glob.glob(os.path.join(outdir, "fig2/*.jpg")))
    for i in tqdm.trange(len(f0)):
        i0 = cv2.imread(f0[i])
        i1 = cv2.imread(f1[i])
        i2 = cv2.imread(f2[i])
        out = np.zeros((2 * i0.shape[0], 2 * i0.shape[1], 3), dtype=np.uint8)
        out[: i0.shape[0], : i0.shape[1]] = i0
        out[: i0.shape[0], i0.shape[1] :] = i1
        out[i0.shape[0] :, : i0.shape[1]] = i2
        cv2.imwrite(os.path.join(outdir, f"{str(i).zfill(4)}.jpg"), out)
