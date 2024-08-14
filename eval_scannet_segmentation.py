import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import scipy
import sklearn.metrics
import torch
import tqdm
import yaml

import clipfusion


labels20 = [
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "couch",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "furniture",
]

prompts20 = []
for label in labels20:
    if label == "other":
        prompts20.append(label)
    else:
        prompts20.append(f"a picture of a {label}")

prompts20[5] = prompts20[5].replace("couch", "sofa")

colors20 = (
    np.array(
        [
            (174, 199, 232),
            (152, 223, 138),
            (31, 119, 180),
            (255, 187, 120),
            (188, 189, 34),
            (140, 86, 75),
            (255, 152, 150),
            (214, 39, 40),
            (197, 176, 213),
            (148, 103, 189),
            (196, 156, 148),
            (23, 190, 207),
            (247, 182, 210),
            (219, 219, 141),
            (255, 127, 14),
            (158, 218, 229),
            (44, 160, 44),
            (112, 128, 144),
            (227, 119, 194),
            (82, 84, 163),
        ]
    ).astype(np.float32)
    / 255
)

labels200 = [
    "wall",
    "chair",
    "floor",
    "table",
    "door",
    "couch",
    "cabinet",
    "shelf",
    "desk",
    "office chair",
    "bed",
    "pillow",
    "sink",
    "picture",
    "window",
    "toilet",
    "bookshelf",
    "monitor",
    "curtain",
    "book",
    "armchair",
    "coffee table",
    "box",
    "refrigerator",
    "lamp",
    "kitchen cabinets",
    "towel",
    "clothes",
    "tv",
    "nightstand",
    "counter",
    "dresser",
    "stool",
    "cushion",
    "plant",
    "ceiling",
    "bathtub",
    "end table",
    "dining table",
    "keyboard",
    "bag",
    "backpack",
    "toilet paper",
    "printer",
    "tv stand",
    "whiteboard",
    "blanket",
    "shower curtain",
    "trash can",
    "closet",
    "stairs",
    "microwave",
    "stove",
    "shoes",
    "computer tower",
    "bottle",
    "bin",
    "ottoman",
    "bench",
    "board",
    "washing machine",
    "mirror",
    "copier",
    "basket",
    "sofa chair",
    "file cabinet",
    "fan",
    "laptop",
    "shower",
    "paper",
    "person",
    "paper towel dispenser",
    "oven",
    "blinds",
    "rack",
    "plate",
    "blackboard",
    "piano",
    "suitcase",
    "rail",
    "radiator",
    "recycling bin",
    "container",
    "wardrobe",
    "soap dispenser",
    "telephone",
    "bucket",
    "clock",
    "stand",
    "light",
    "laundry basket",
    "pipe",
    "clothes dryer",
    "guitar",
    "toilet paper holder",
    "seat",
    "speaker",
    "column",
    "bicycle",
    "ladder",
    "bathroom stall",
    "shower wall",
    "cup",
    "jacket",
    "storage bin",
    "coffee maker",
    "dishwasher",
    "paper towel roll",
    "machine",
    "mat",
    "windowsill",
    "bar",
    "toaster",
    "bulletin board",
    "ironing board",
    "fireplace",
    "soap dish",
    "kitchen counter",
    "doorframe",
    "toilet paper dispenser",
    "mini fridge",
    "fire extinguisher",
    "ball",
    "hat",
    "shower curtain rod",
    "water cooler",
    "paper cutter",
    "tray",
    "shower door",
    "pillar",
    "ledge",
    "toaster oven",
    "mouse",
    "toilet seat cover dispenser",
    "furniture",
    "cart",
    "storage container",
    "scale",
    "tissue box",
    "light switch",
    "crate",
    "power outlet",
    "decoration",
    "sign",
    "projector",
    "closet door",
    "vacuum cleaner",
    "candle",
    "plunger",
    "stuffed animal",
    "headphones",
    "dish rack",
    "broom",
    "guitar case",
    "range hood",
    "dustpan",
    "hair dryer",
    "water bottle",
    "handicap bar",
    "purse",
    "vent",
    "shower floor",
    "water pitcher",
    "mailbox",
    "bowl",
    "paper bag",
    "alarm clock",
    "music stand",
    "projector screen",
    "divider",
    "laundry detergent",
    "bathroom counter",
    "object",
    "bathroom vanity",
    "closet wall",
    "laundry hamper",
    "bathroom stall door",
    "ceiling light",
    "trash bin",
    "dumbbell",
    "stair rail",
    "tube",
    "bathroom cabinet",
    "cd case",
    "closet rod",
    "coffee kettle",
    "structure",
    "shower head",
    "keyboard piano",
    "case of water bottles",
    "coat rack",
    "storage organizer",
    "folded chair",
    "fire alarm",
    "power strip",
    "calendar",
    "poster",
    "potted plant",
    "luggage",
    "mattress",
]

prompts200 = labels200.copy()
prompts200[5] = "sofa"

colors200 = (
    np.array(
        [
            [174, 199, 232],
            [188, 189, 34],
            [152, 223, 138],
            [255, 152, 150],
            [214, 39, 40],
            [91, 135, 229],
            [31, 119, 180],
            [229, 91, 104],
            [247, 182, 210],
            [91, 229, 110],
            [255, 187, 120],
            [141, 91, 229],
            [112, 128, 144],
            [196, 156, 148],
            [197, 176, 213],
            [44, 160, 44],
            [148, 103, 189],
            [229, 91, 223],
            [219, 219, 141],
            [192, 229, 91],
            [88, 218, 137],
            [58, 98, 137],
            [177, 82, 239],
            [255, 127, 14],
            [237, 204, 37],
            [41, 206, 32],
            [62, 143, 148],
            [34, 14, 130],
            [143, 45, 115],
            [137, 63, 14],
            [23, 190, 207],
            [16, 212, 139],
            [90, 119, 201],
            [125, 30, 141],
            [150, 53, 56],
            [186, 197, 62],
            [227, 119, 194],
            [38, 100, 128],
            [120, 31, 243],
            [154, 59, 103],
            [169, 137, 78],
            [143, 245, 111],
            [37, 230, 205],
            [14, 16, 155],
            [196, 51, 182],
            [237, 80, 38],
            [138, 175, 62],
            [158, 218, 229],
            [38, 96, 167],
            [190, 77, 246],
            [208, 49, 84],
            [208, 193, 72],
            [55, 220, 57],
            [10, 125, 140],
            [76, 38, 202],
            [191, 28, 135],
            [211, 120, 42],
            [118, 174, 76],
            [17, 242, 171],
            [20, 65, 247],
            [208, 61, 222],
            [162, 62, 60],
            [210, 235, 62],
            [45, 152, 72],
            [35, 107, 149],
            [160, 89, 237],
            [227, 56, 125],
            [169, 143, 81],
            [42, 143, 20],
            [25, 160, 151],
            [82, 75, 227],
            [253, 59, 222],
            [240, 130, 89],
            [123, 172, 47],
            [71, 194, 133],
            [24, 94, 205],
            [134, 16, 179],
            [159, 32, 52],
            [213, 208, 88],
            [64, 158, 70],
            [18, 163, 194],
            [65, 29, 153],
            [177, 10, 109],
            [152, 83, 7],
            [83, 175, 30],
            [18, 199, 153],
            [61, 81, 208],
            [213, 85, 216],
            [170, 53, 42],
            [161, 192, 38],
            [23, 241, 91],
            [12, 103, 170],
            [151, 41, 245],
            [133, 51, 80],
            [184, 162, 91],
            [50, 138, 38],
            [31, 237, 236],
            [39, 19, 208],
            [223, 27, 180],
            [254, 141, 85],
            [97, 144, 39],
            [106, 231, 176],
            [12, 61, 162],
            [124, 66, 140],
            [137, 66, 73],
            [250, 253, 26],
            [55, 191, 73],
            [60, 126, 146],
            [153, 108, 234],
            [184, 58, 125],
            [135, 84, 14],
            [139, 248, 91],
            [53, 200, 172],
            [63, 69, 134],
            [190, 75, 186],
            [127, 63, 52],
            [141, 182, 25],
            [56, 144, 89],
            [64, 160, 250],
            [182, 86, 245],
            [139, 18, 53],
            [134, 120, 54],
            [49, 165, 42],
            [51, 128, 133],
            [44, 21, 163],
            [232, 93, 193],
            [176, 102, 54],
            [116, 217, 17],
            [54, 209, 150],
            [60, 99, 204],
            [129, 43, 144],
            [252, 100, 106],
            [187, 196, 73],
            [13, 158, 40],
            [52, 122, 152],
            [128, 76, 202],
            [187, 50, 115],
            [180, 141, 71],
            [77, 208, 35],
            [72, 183, 168],
            [97, 99, 203],
            [172, 22, 158],
            [155, 64, 40],
            [118, 159, 30],
            [69, 252, 148],
            [45, 103, 173],
            [111, 38, 149],
            [184, 9, 49],
            [188, 174, 67],
            [53, 206, 53],
            [97, 235, 252],
            [66, 32, 182],
            [236, 114, 195],
            [241, 154, 83],
            [133, 240, 52],
            [16, 205, 144],
            [75, 101, 198],
            [237, 95, 251],
            [191, 52, 49],
            [227, 254, 54],
            [49, 206, 87],
            [48, 113, 150],
            [125, 73, 182],
            [229, 32, 114],
            [158, 119, 28],
            [60, 205, 27],
            [18, 215, 201],
            [79, 76, 153],
            [134, 13, 116],
            [192, 97, 63],
            [108, 163, 18],
            [95, 220, 156],
            [98, 141, 208],
            [144, 19, 193],
            [166, 36, 57],
            [212, 202, 34],
            [23, 206, 34],
            [91, 211, 236],
            [79, 55, 137],
            [182, 19, 117],
            [134, 76, 14],
            [87, 185, 28],
            [82, 224, 187],
            [92, 110, 214],
            [168, 80, 171],
            [197, 63, 51],
            [175, 199, 77],
            [62, 180, 98],
            [8, 91, 150],
            [77, 15, 130],
            [154, 65, 96],
            [197, 152, 11],
            [59, 155, 45],
            [12, 147, 145],
            [54, 35, 219],
            [210, 73, 181],
            [221, 124, 77],
            [149, 214, 66],
            [72, 185, 134],
            [42, 94, 198],
        ]
    ).astype(np.float32)
    / 255
)


def get_gt_labels(scan_dir, classes="20"):
    scan_name = os.path.basename(scan_dir)

    aggfile = os.path.join(scan_dir, f"{scan_name}.aggregation.json")
    segfile = os.path.join(scan_dir, f"{scan_name}_vh_clean_2.0.010000.segs.json")

    # aggfile = os.path.join(scan_dir, f"{scan_name}_vh_clean.aggregation.json")
    # segfile = os.path.join(scan_dir, f"{scan_name}_vh_clean.segs.json")

    with open(aggfile, "r") as f:
        agg = json.load(f)

    with open(segfile, "r") as f:
        segs = json.load(f)

    nverts = len(segs["segIndices"])

    segments = {}
    for group in agg["segGroups"]:
        for seg_idx in group["segments"]:
            segments[seg_idx] = group["label"]

    if classes == "20":
        class_to_color = {clas: tuple(color) for color, clas in zip(colors20, labels20)}
        class_to_idx = {clas: i for i, clas in enumerate(labels20)}
    elif classes == "200":
        class_to_color = {
            clas: tuple(color) for color, clas in zip(colors200, labels200)
        }
        class_to_idx = {clas: i for i, clas in enumerate(labels200)}
    else:
        raise NotImplementedError

    extra = []
    vertex_labels = np.full((nverts,), -1, dtype=np.int32)
    for vert_idx, seg_idx in enumerate(segs["segIndices"]):
        if seg_idx in segments:
            category = segments[seg_idx]

            if category == "sofa":
                raise Exception("gah. is this an alias for couch?")

            if category in class_to_color:
                vertex_labels[vert_idx] = class_to_idx[category]
            else:
                extra.append(category)
    # if extra:
    #     print(np.unique(extra))
    #     raise Exception()

    return vertex_labels


def segment(clip, vertex_feat_file, prompts):
    vertex_clip_feats = torch.from_numpy(np.load(vertex_feat_file))

    feat_norm = vertex_clip_feats.norm(dim=-1, keepdim=True)
    feat_norm.clamp_min_(0.1)
    vertex_clip_feats /= feat_norm

    if vertex_clip_feats.isnan().any():
        raise Exception("found nans")

    clip_text_features = clip.text_inference(prompts)
    dotprod = vertex_clip_feats @ clip_text_features.T
    relevance = (100 * dotprod).softmax(dim=-1)

    vertex_labels = relevance.argsort(dim=-1, descending=True)
    return vertex_labels


def eval_scene(pred_dir, gt_dir, classes, clip):
    if classes == "20":
        labels = labels20
        prompts = prompts20
        colors = colors20
    elif classes == "200":
        labels = labels200
        prompts = prompts200
        colors = colors200
    else:
        raise NotImplementedError

    vertex_feat_file = os.path.join(pred_dir, "vertex_clip_feats.npy")
    pred_meshfile = os.path.join(pred_dir, "mesh_rgb.ply")
    gt_meshfile = os.path.join(gt_dir, f"{os.path.basename(gt_dir)}_vh_clean_2.ply")

    pred_mesh = o3d.io.read_triangle_mesh(pred_meshfile)
    gt_mesh = o3d.io.read_triangle_mesh(gt_meshfile)

    pred_vertex_labels = segment(clip, vertex_feat_file, prompts)
    gt_vertex_labels = get_gt_labels(gt_dir, classes=classes)
    kdt = scipy.spatial.KDTree(pred_mesh.vertices)
    _, inds = kdt.query(gt_mesh.vertices)
    transferred_vertex_labels = pred_vertex_labels[inds]

    correct_top5 = torch.any(
        torch.from_numpy(gt_vertex_labels)[:, None] == transferred_vertex_labels[:, :5],
        dim=-1,
    )
    correct_top1 = torch.from_numpy(gt_vertex_labels) == transferred_vertex_labels[:, 0]
    ncorrect_top1 = []
    ncorrect_top5 = []
    ntotal = []
    for i in range(len(labels)):
        vertex_mask = gt_vertex_labels == i
        ncorrect_top5.append(correct_top5[vertex_mask].sum().item())
        ncorrect_top1.append(correct_top1[vertex_mask].sum().item())
        ntotal.append(vertex_mask.sum().item())

    np.save(
        os.path.join(pred_dir, "transferred_vertex_labels.npy"),
        transferred_vertex_labels,
    )
    np.save(os.path.join(pred_dir, "gt_vertex_labels.npy"), gt_vertex_labels)

    gt_vertex_colors = np.zeros((len(gt_vertex_labels), 3), dtype=np.float32)
    idx = gt_vertex_labels > -1
    gt_vertex_colors[idx] = colors[gt_vertex_labels[idx]]
    transferred_vertex_colors = colors[transferred_vertex_labels[:, 0]]
    transferred_vertex_colors[~idx] = 0

    segmented_mesh = o3d.geometry.TriangleMesh(pred_mesh)
    transferred_mesh = o3d.geometry.TriangleMesh(gt_mesh)

    gt_mesh.vertex_colors = o3d.utility.Vector3dVector(gt_vertex_colors)
    segmented_mesh.vertex_colors = o3d.utility.Vector3dVector(
        colors[pred_vertex_labels[:, 0]]
    )
    transferred_mesh.vertex_colors = o3d.utility.Vector3dVector(
        transferred_vertex_colors
    )

    valid_vert_idx = np.all(gt_vertex_colors != 0, axis=-1)
    correct_idx = np.all(
        (transferred_vertex_colors == gt_vertex_colors)[:, :3], axis=-1
    )
    correct_colors = plt.cm.jet(correct_idx.astype(np.float32))[:, :3]
    correct_colors[~valid_vert_idx] = 0

    correct_mesh = o3d.geometry.TriangleMesh(gt_mesh)
    correct_mesh.vertex_colors = o3d.utility.Vector3dVector(correct_colors)

    o3d.io.write_triangle_mesh(os.path.join(pred_dir, "gt.ply"), gt_mesh)
    o3d.io.write_triangle_mesh(os.path.join(pred_dir, "segmented.ply"), segmented_mesh)
    o3d.io.write_triangle_mesh(
        os.path.join(pred_dir, "transferred.ply"), transferred_mesh
    )
    o3d.io.write_triangle_mesh(os.path.join(pred_dir, "correct.ply"), correct_mesh)

    # correct_top1 = transferred_vertex_labels[:, 0].numpy() == gt_vertex_labels
    # correct_top5 = np.any(transferred_vertex_labels[:, :5].numpy() == gt_vertex_labels[:, None], axis=-1)

    # top1_acc_overall = np.mean(correct_top1)
    # top5_acc_overall = np.mean(correct_top5)

    # per_class_top1_acc = np.empty(len(labels))
    # per_class_top5_acc = np.empty(len(labels))
    # for i in range(len(labels)):
    #     per_class_top1_acc[i] = np.mean(correct_top1[gt_vertex_labels == i])
    #     per_class_top5_acc[i] = np.mean(correct_top5[gt_vertex_labels == i])

    cmat = sklearn.metrics.confusion_matrix(
        gt_vertex_labels,
        transferred_vertex_labels[:, 0],
        labels=list(range(len(labels))),
    )

    # print(",".join([str(i) for i in line]))
    return cmat, ncorrect_top1, ncorrect_top5, ntotal


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_dir")
    parser.add_argument("gt_dir")
    parser.add_argument("--scan-name", help="restrict to a single scan")
    parser.add_argument("--classes", "-c", default="20", choices=["20", "200"])
    args = parser.parse_args()

    pred_dirs = [
        d
        for d in sorted(glob.glob(os.path.join(args.pred_dir, "scene*")))
        if os.path.isdir(d)
    ]
    gt_dirs = sorted(glob.glob(os.path.join(args.gt_dir, "scene*")))

    if args.scan_name is not None:
        for pred_dir in pred_dirs:
            if os.path.basename(pred_dir) == args.scan_name:
                pred_dirs = [pred_dir]
                print(f"limiting to scan: {args.scan_name}")
                break
        else:
            raise Exception(f"couldn't find a scan called {args.scan_name}")

    global_cmat = 0
    scene_cmats = {}
    ncorrect_top1 = 0
    ncorrect_top5 = 0
    ntotal = 0

    for pred_dir in tqdm.tqdm(pred_dirs):
        scene_name = os.path.basename(pred_dir)
        for gt_dir in gt_dirs:
            if scene_name in gt_dir:
                break
        else:
            raise Exception(f"couldn't find gt_dir for scene: {scene_name}")

        assert os.path.basename(pred_dir) == os.path.basename(gt_dir)

        config_file = os.path.join(pred_dir, "config.yml")
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        clip = clipfusion.Clip(config["clip_model"], config["clip_pretraining"])
        clip.requires_grad_(False)
        clip.eval()

        scene_cmat, _ncorrect_top1, _ncorrect_top5, _ntotal = eval_scene(
            pred_dir, gt_dir, args.classes, clip
        )
        scene_cmats[scene_name] = scene_cmat.tolist()

        global_cmat += scene_cmat
        ncorrect_top1 += np.array(_ncorrect_top1)
        ncorrect_top5 += np.array(_ncorrect_top5)
        ntotal += np.array(_ntotal)

    tp = np.diagonal(global_cmat)
    fn = np.sum(global_cmat, axis=-1) - tp
    fp = np.sum(global_cmat, axis=0) - tp
    iou = tp / (tp + fp + fn)
    miou = np.nanmean(iou)

    acc_top1 = ncorrect_top1 / ntotal
    acc_top5 = ncorrect_top5 / ntotal

    mAcc_top1 = np.nanmean(acc_top1)
    mAcc_top5 = np.nanmean(acc_top5)

    print(np.round(100 * miou, 1))
    print(np.round(100 * mAcc_top1, 1))
    print(np.round(100 * mAcc_top5, 1))

    for i in iou:
        print(np.round(100 * i, 1))
    # for i in acc_top1:
    #     print(np.round(100 * i, 1))
    # for i in acc_top5:
    #     print(np.round(100 * i, 1))

    with open(os.path.join(args.pred_dir, "scene_cmats.json"), "w") as f:
        json.dump(scene_cmats, f)

    np.save(os.path.join(args.pred_dir, f"global_cmat.npy"), global_cmat)
