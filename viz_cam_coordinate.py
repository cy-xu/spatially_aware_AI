import glob
import os
import json

import numpy as np
import trimesh

scene_dir = "scenes/iphone_3dscanner/5110_my_corner"

posefiles = sorted(glob.glob(os.path.join(scene_dir, 'frame_*.json')))

poses = []
for posefile in posefiles:
    with open(posefile, 'r') as f:
        meta = json.load(f)
        pose = np.array(meta['cameraPoseARFrame']).reshape(4, 4)
        poses.append(pose)

poses = np.array(poses)

_ = trimesh.PointCloud(poses[:1, :3, 3]).export('cam.ply')
_ = trimesh.PointCloud(poses[:1, :3, 3] + .1 * poses[:1, :3, 0]).export('cam_x.ply')
_ = trimesh.PointCloud(poses[:1, :3, 3] + .1 * poses[:1, :3, 1]).export('cam_y.ply')
_ = trimesh.PointCloud(poses[:1, :3, 3] + .1 * poses[:1, :3, 2]).export('cam_z.ply')
