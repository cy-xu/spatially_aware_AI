import os
import sys
import numpy as np
from dgcnn.data import InSituVoxelData
import open3d as o3d
import torch

# add kmax-deeplab to sys path
sys.path.append(os.path.join(os.getcwd(), "kmax"))

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.visualizer import ColorMode, Visualizer, _PanopticPrediction
from detectron2.modeling import build_model
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T

from kmax.kmax_deeplab import add_kmax_deeplab_config
from kmax.constants import COCO_PANOPTIC_CLASSES, COCO_PANOPTIC_COLORS

# constants for the 133 stuff classes
predefined_classes = [
    name.replace("-other", "").replace("-merged", "") for name in COCO_PANOPTIC_CLASSES
] + ["others"]
predefined_colors = COCO_PANOPTIC_COLORS + [[0, 0, 0]]


class KmaxSegmentationModel:
    def __init__(self, config_file, weight_path, device="cpu"):
        self.cfg = get_cfg()
        add_deeplab_config(self.cfg)
        add_kmax_deeplab_config(self.cfg)
        self.cfg.merge_from_file(config_file)
        self.cfg.freeze()

        self.metadata = MetadataCatalog.get(
            self.cfg.DATASETS.TEST[0] if len(self.cfg.DATASETS.TEST) else "__unused"
        )
        self.device = torch.device(device)
        self.cpu_device = torch.device("cpu")
        self.instance_mode = ColorMode.IMAGE

        self.model = build_model(self.cfg)
        self.model.eval()

        if len(self.cfg.DATASETS.TEST):
            metadata = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(weight_path)

        self.aug = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST],
            self.cfg.INPUT.MAX_SIZE_TEST,
        )

        self.input_format = self.cfg.INPUT.FORMAT

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        # predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.

        # https://github.com/sphinx-doc/sphinx/issues/4258
        with torch.no_grad():
            _, height, width = image.shape

            # resize long edge to 1281, keep aspect ratio
            aspect_ratio = width / height
            if aspect_ratio > 1:
                new_width = 1281
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = 1281
                new_width = int(new_height * aspect_ratio)

            image = torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=(new_height, new_width),
                mode="bilinear",
                align_corners=False,
            )

            # input is RGB tensor [3, h, w], convert to BGR
            image = image[0]
            image = image[[2, 1, 0], :, :]
            # input is normalized to [0, 1], convert to [0, 255] torch int32
            image = image * 255
            image = image.to(torch.int32)

            inputs = {"image": image, "height": height, "width": width}

            predictions = self.model([inputs])[0]

        panoptic_seg, segments_info = predictions["panoptic_seg"]

        # fill the mask with segmentation category_id
        obj_idx_mask = predictions["panoptic_seg"][0].clone()

        # things_sum = len(self.metadata.thing_classes) # 80 total
        # stuff_sum = len(self.metadata.stuff_classes) # 133 total

        # make empty mask (0s) to total class num to avoid confusing with person
        obj_idx_mask[obj_idx_mask == 0] = 133

        pred = _PanopticPrediction(
            panoptic_seg.to(self.cpu_device), segments_info, self.metadata
        )

        # draw mask for all semantic segments first i.e. "stuff"
        for mask, sinfo in pred.semantic_masks():
            category_idx = sinfo["category_id"]

            # the above essentially does this for me
            obj_idx_mask[mask] = category_idx

        # draw mask for all instances second
        all_instances = list(pred.instance_masks())

        if len(all_instances) == 0:
            return obj_idx_mask

        masks, sinfo = list(zip(*all_instances))
        for i, info in enumerate(sinfo):
            obj_idx_mask[masks[i]] = info["category_id"]

        ## TODO: use instance information to separate objects of the same class

        # category_ids = [x["category_id"] for x in sinfo]

        # try:
        #     scores = [x["score"] for x in sinfo]
        # except KeyError:
        #     scores = None
        # labels = _create_text_labels(
        #     category_ids,
        #     scores,
        #     self.metadata.thing_classes,
        #     [x.get("iscrowd", 0) for x in sinfo],
        # )

        # try:
        #     colors = [
        #         self._jitter([x / 255 for x in self.metadata.thing_colors[c]])
        #         for c in category_ids
        #     ]
        # except AttributeError:
        #     colors = None
        # self.overlay_instances(
        #     masks=masks, labels=labels, assigned_colors=colors, alpha=alpha
        # )

        return obj_idx_mask


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_path(config: dotdict, curr_ver: int, key=None) -> dotdict:
    """
    Get or set the path for various files
    """
    ver = f"v{curr_ver:02d}"

    paths = {
        "scene_inputdir": os.path.join(config.scan_dir, config.scan_name, ver),
        "scene_dir": os.path.join(config.output_dir, config.scan_name),
        "scene_outputdir": os.path.join(config.output_dir, config.scan_name, ver),
        # shared between versions
        "insitu_model_path": os.path.join(
            config.output_dir, config.scan_name, "insitu_model.pth"
        ),
        "insitu_labels": os.path.join(
            config.output_dir, config.scan_name, "insitu_labels.json"
        ),
        # version based
        "scene_knowledge": os.path.join(
            config.output_dir, config.scan_name, ver, "scene_knowledge.json"
        ),
        "vertex_clip_feats": os.path.join(
            config.output_dir, config.scan_name, ver, "vertex_clip_feats.npy"
        ),
        "vertex_obj_idx": os.path.join(
            config.output_dir, config.scan_name, ver, "vertex_obj_idx.npy"
        ),
        "voxel_clip_feats": os.path.join(
            config.output_dir, config.scan_name, ver, "voxel_clip_feats.npy"
        ),
        "voxel_rgb": os.path.join(
            config.output_dir, config.scan_name, ver, "voxel_rgb.npy"
        ),
        "mesh_rgb": os.path.join(
            config.output_dir, config.scan_name, ver, "mesh_rgb.ply"
        ),
        "mesh_segmentation": os.path.join(
            config.output_dir, config.scan_name, ver, "mesh_segmentation.ply"
        ),
    }

    if key is None:
        for key, path in paths.items():
            setattr(config, key, path)
        return config

    elif key in paths:
        return paths[key]
    else:
        raise ValueError(f"invalid key: {key}")


def mesh_to_json(config, version, mesh_type):
    mesh_path = get_path(config, version, mesh_type)
    print(f'loading mesh from "{mesh_path}')

    # load mesh file with o3d
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    verts = np.asarray(mesh.vertices).astype(np.float32)
    faces = np.asarray(mesh.triangles)  # int32
    colors = np.asarray(mesh.vertex_colors).astype(np.float32)  # float64 0-1

    # convert to json
    mesh_json = {
        "vertices": verts.tolist(),
        "faces": faces.tolist(),
        "colors": colors.tolist(),
    }

    return mesh_json


def add_object(
    unique_objects,
    object_counts,
    gt_labels,
    object_index,
    class_id,
    class_label,
    curr_voxels,
    user_modified=False,
    merged=False,
):
    # get current count, if not found, return 0 + 1
    obj_id, class_label = get_obj_counts(object_counts, class_label)

    # if merge_this:
    #     # if the object is merged, we add count number to its name
    #     obj_id = class_label
    #     # if the object already exists, just update the voxels
    #     if class_label in unique_objects:
    #         unique_objects[obj_id]["voxels"] += curr_voxels
    #         return obj_id, unique_objects, object_counts, gt_labels
    # else:
    #     obj_id = f"{class_label}_{object_counts[class_label]}"
    #     obj_id = new_label

    # Training data ground truth labels are defined here
    if user_modified:
        if obj_id not in gt_labels:
            # add new label to ground truth labels
            gt_labels.append(obj_id)

    color = predefined_colors[class_id]

    # if the object is not in the dict, add it
    unique_objects[obj_id] = {
        "class_id": class_id,
        "class_label": class_label,
        "voxels": curr_voxels,
        "object_index": object_index,
        "gt_label": obj_id,
        "user_modified": user_modified,
        "merged": merged,
        "removed": False,
        "color": color,
    }

    # 'index_pointer': object_index,

    return obj_id


def flood_fill_3d(
    array_3d,
    scene_knowledge,
    voxel_clip_feats,
    voxel_rgb,
    insitu_model,
    scene_knowledge_prev=None,
):
    """
    :param array_3d: np.array, 3d numpy array, each voxel is a class ID
    :param predefined_classes: list, list of class names
    :param scene_knowledge: dict, all knowledge of the scene
    :param voxel_clip_feats: np.array, clip features for each voxel
    :param insitu_model: InSituLearning, the in-situ model

    :return: scene_knowledge: dict, updated all knowledge of the scene
    """

    def get_neighbors(coord):
        x, y, z = coord
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == dy == dz == 0:
                        continue
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if (
                        0 <= nx < array_3d.shape[0]
                        and 0 <= ny < array_3d.shape[1]
                        and 0 <= nz < array_3d.shape[2]
                    ):
                        neighbors.append((nx, ny, nz))
        return neighbors

    def flood_fill(coord, label):
        stack = [coord]
        visited = set()
        object_voxels = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            x, y, z = current
            if array_3d[x, y, z] == label:
                object_voxels.append(current)
                neighbors = get_neighbors(current)
                stack.extend(neighbors)
        return object_voxels

    # keep track of visited voxels to avoid redundant computation
    visited_voxels = set()
    voxel_obj_ids = np.ones(array_3d.shape, dtype=np.int32) * -1

    # keep track of all unique objects in the scene
    unique_objects = {}
    # another dict to count repeated objects of the same class
    object_counts = {}
    negative_object_index = -2

    # we use three lists to keep track of possible states of an object
    unchanged_objects = {}
    new_objects = {}
    missing_objects = {}
    # freeze the labels so newly added stuff won't add to missing
    labels_freezed = insitu_model.labels[1:].copy()

    # debug
    print(f"\nUsing in-situ model for classification: {insitu_model.model_trained:}")

    for x in range(array_3d.shape[0]):
        for y in range(array_3d.shape[1]):
            for z in range(array_3d.shape[2]):
                class_id = array_3d[x, y, z].item()
                class_label = predefined_classes[class_id]
                user_modified = False

                if (x, y, z) in visited_voxels:
                    continue
                visited_voxels.update([(x, y, z)])

                # null class is 133, empty voxels are -1
                # cat_id 0 is person
                if class_id == 133 or class_id == -1:
                    continue

                # 3d flood fill to get all voxels of the same object
                curr_voxels = flood_fill((x, y, z), class_id)
                vox_count = len(curr_voxels)

                # reject small objects less than 3 voxels
                if vox_count < 3:
                    visited_voxels.update(curr_voxels)
                    continue

                # index voxels with object_voxels
                vox_indices = tuple(zip(*curr_voxels))
                # the object index used in the voxel grid
                object_index = negative_object_index

                if insitu_model.model_trained:
                    # if in-situ model exists, check if this object existed
                    # this ensure repeated scans produce the same object ID

                    object_features = {
                        "clip_feats": voxel_clip_feats[vox_indices],
                        "rgb": voxel_rgb[vox_indices],
                        "voxels": np.array(curr_voxels),
                    }

                    all_features = InSituVoxelData.preprocess(
                        [object_features], None, inference=True
                    )

                    # model predict a matching object index from previous scan
                    pred_label_index = insitu_model.predict(all_features)

                    # if the object is found in the previous scan
                    # 0 would be the null class
                    if pred_label_index > 0:
                        # we use the previously user defined label
                        class_label = insitu_model.labels[pred_label_index]
                        user_modified = True

                        print(
                            f"found: {predefined_classes[class_id]}, model predicted: {class_label}"
                        )

                        # object index voxel grid uses gt_label index
                        object_index = pred_label_index

                obj_id = add_object(
                    unique_objects,
                    object_counts,
                    insitu_model.labels,
                    object_index,
                    class_id,
                    class_label,
                    curr_voxels,
                    user_modified=user_modified,
                    merged="merged" in class_label,
                )

                if insitu_model.model_trained > 0:
                    if pred_label_index > 0:
                        # if the object is found in the previous scan
                        # we add it to unchanged list
                        unchanged_objects[obj_id] = unique_objects[obj_id]
                        print(f"object {obj_id} is unchanged in the new scan")

                # end of an object
                visited_voxels.update(curr_voxels)
                voxel_obj_ids[vox_indices] = object_index
                # only decrement if the index is negative
                if object_index < 0:
                    negative_object_index -= 1

    # check if any object is missing, if previous knowledge exists
    if scene_knowledge_prev:
        for gt_label in labels_freezed:
            if gt_label not in unique_objects.keys():
                missing_objects[gt_label] = scene_knowledge_prev["unique_objects"][
                    gt_label
                ]
                print(f"object {gt_label} is missing in the new scan")

    # print("We found these unique objects:")
    # for obj_id, obj_value in unique_objects.items():
    #     print(f"{obj_id}: size {len(obj_value['voxels'])} voxels")

    # collect all knowlege of the scene
    if scene_knowledge is None:
        scene_knowledge = {}

    # unique_objects is {obj_id: {obj_properties}}
    scene_knowledge["unique_objects"] = unique_objects
    # object_counts is {class_label: count}
    scene_knowledge["object_counts"] = object_counts

    # same as unique_objects: {obj_id: {obj_properties}}
    scene_knowledge["unchanged_objects"] = unchanged_objects
    scene_knowledge["new_objects"] = new_objects
    scene_knowledge["missing_objects"] = missing_objects

    return scene_knowledge, voxel_obj_ids


def get_obj_counts(object_counts, obj_id):
    # check if the obj_id has index number already
    if ":" in obj_id:
        possible_label, possible_int = obj_id.split(":")[0], obj_id.split(":")[-1]

        # then confirm if the last part is an int
        if possible_int.isdigit():
            class_label = possible_label
    else:
        class_label = obj_id

    # get current count, if not found, return 0 + 1
    object_counts[class_label] = object_counts.get(class_label, 0) + 1
    id_with_idx = f"{class_label}:{object_counts[class_label]}"

    return id_with_idx, class_label


def mark_object_of_interest(scene_knowledge, insitu_model, object_list):
    # mark objects in the list as user_modified so ML can learn

    if len(object_list) < 1:
        print("Not enough objects to memorize")
        return scene_knowledge

    unique_objects = scene_knowledge["unique_objects"]

    for obj_id in object_list:
        if obj_id in unique_objects:
            unique_objects[obj_id]["user_modified"] = True

            # add new label to ground truth labels
            if obj_id not in insitu_model.labels:
                insitu_model.labels.append(obj_id)

            unique_objects[obj_id]["gt_label"] = obj_id

        else:
            print(f"object {obj_id} not found")

    return scene_knowledge


def merge_objects(scene_knowledge, vertex_obj_idx, insitu_model, merge_list, new_label):
    # here the intput is the updated unique_objects dict
    # we update the insitu model with the changes

    if len(merge_list) < 1:
        print("Not enough objects to merge or rename")
        return scene_knowledge

    # add "-merged" to the new label so ML model's prediction can be merged too
    if len(merge_list) > 1 and "merged" not in new_label:
        new_label = f"{new_label}-merged"

    unique_objects = scene_knowledge["unique_objects"]
    object_counts = scene_knowledge["object_counts"]

    # get current count, if not found, return 0 + 1
    new_label, class_label = get_obj_counts(object_counts, new_label)

    # add new label to ground truth labels
    if new_label not in insitu_model.labels:
        insitu_model.labels.append(new_label)

    obj_index = insitu_model.labels.index(new_label)

    # maek a copy of the first in the list
    target_object = unique_objects[merge_list[0]].copy()
    # obj_index_prev = target_object["object_index"]

    # previous_verts_count = np.sum(vertex_obj_idx == obj_index_prev)
    # print(
    #     f"preivous object index: {obj_index_prev}, verts count: {previous_verts_count}"
    # )

    target_object["merged"] = len(merge_list) > 1
    target_object["user_modified"] = True
    target_object["gt_label"] = new_label
    target_object["class_label"] = class_label
    target_object["object_index"] = obj_index

    # # update vertex_obj_idx for mesh extraction
    # vertex_obj_idx[vertex_obj_idx == obj_index_prev] = obj_index

    # then merge the voxels
    for i, obj_id in enumerate(merge_list):
        if i == 0:
            del unique_objects[obj_id]
            continue
        target_object["voxels"] += unique_objects[obj_id]["voxels"]
        # obj_idx_merge = unique_objects[obj_id]["object_index"]
        # vertex_obj_idx[vertex_obj_idx == obj_idx_merge] = obj_index
        del unique_objects[obj_id]

    # update the scene knowledge
    unique_objects[new_label] = target_object
    scene_knowledge["unique_objects"] = unique_objects

    return new_label, scene_knowledge


def extract_mesh_by_object(vertices, faces, colors, vertex_indices, obj_idx):
    # Find indices of vertices for the given object_id
    object_indices = np.where(vertex_indices == obj_idx)[0]

    # print(f"object {obj_idx} has {len(object_indices)} vertices")

    # Select vertices and colors by these indices
    object_vertices = vertices[object_indices]
    object_colors = colors[object_indices]

    # To select faces, we need to find faces in which all of its vertices are belong to object
    filter_faces = np.isin(faces, object_indices).all(axis=1)
    object_faces = faces[filter_faces]

    # Reindex faces for the new vertices
    map_indices = {v: i for i, v in enumerate(object_indices)}
    for face in object_faces:
        for i in range(3):
            face[i] = map_indices[face[i]]

    # Convert arrays to open3d format and return
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(object_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(object_faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(object_colors)

    return object_vertices, object_faces, object_colors, mesh
