# visulize the 3d voxel grid with matplotlib 3d and export to x3d

import os
import json
import numpy as np
import matplotlib.pyplot as plt

# read "color_objects.npy" and "unique_objects.json"
scene_outputdir = "./fusion_output/scene0000_00"
color_objects = np.load(os.path.join(scene_outputdir, "color_objects.npy"), allow_pickle=True)
# grid_objects is whereever the color_objects is not None
grid_objects = color_objects != None

unique_objects = json.load(open(os.path.join(scene_outputdir, "unique_objects.json"), "r"))

# plot 3d voxel grid with matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.voxels(category_voxels.cpu().numpy(), facecolors=rgb_colors, alpha=0.5)
# occupancy_grid.shape (80, 87, 26)
# rgb_objects.shape (80, 87, 26, 3)
ax.voxels(grid_objects, facecolors=color_objects)
ax.view_init(elev=45, azim=90)  # Adjust azimuth as needed
plt.show()
# plt.savefig(os.path.join(scene_outputdir, "occupancy_grid_elev45_azim90.png"))
# convert to x3d

