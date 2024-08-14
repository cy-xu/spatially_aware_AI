import os
import numpy as np
import open3d as o3d

# Assume following:
# vertices is an Nx3 numpy array
# faces is an Mx3 numpy array
# colors is an Nx3 numpy array
# labels is an N sized numpy array with object ids for each vertex


def extract_mesh_by_id(vertices, faces, colors, labels, object_id):
    # Find indices of vertices for the given object_id
    object_indices = np.where(labels == object_id)[0]

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

    return mesh


# load a mesh
input_dir = "unity_output/5110_my_corner/v01/"
out_dir = input_dir + "object_meshes_vis/"
os.makedirs(out_dir, exist_ok=True)

obj_idx = np.load(input_dir + "vertex_obj_idx.npy")
unique_objects = np.unique(obj_idx)
print(f"unique object ids: {unique_objects}")


mesh_rgb = o3d.io.read_triangle_mesh(input_dir + "mesh_rgb.ply")
# extract mesh by object id
vertices = np.asarray(mesh_rgb.vertices)
faces = np.asarray(mesh_rgb.triangles)
colors = np.asarray(mesh_rgb.vertex_colors)


for obj_id in unique_objects:
    mesh = extract_mesh_by_id(vertices, faces, colors, obj_idx, obj_id)
    o3d.io.write_triangle_mesh(out_dir + f"object_{obj_id}_mesh.ply", mesh)
