import numpy as np
import requests
import json
import open3d as o3d

# server = "http://127.0.0.1:3291/"
headers = {"Content-type": "application/x-www-form-urlencoded"}


def draw_a_mesh(verts, faces, colors, name):
    # if color has four channels, remove the last one
    colors = np.array(colors)
    if colors.shape[1] == 4:
        colors = colors[:, :3]

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_triangle_mesh(f"client_requested_mesh/{name}_mesh.ply", mesh)


def test_text_query(text="show me a comfortable place to sit"):
    data = {"text_query": text}
    response = requests.post(server + "text_query", data=data)
    print(f"requesting mesh for text query: {text}")

    if response.ok:
        response_json = json.loads(response.text)
        draw_a_mesh(
            response_json["vertices"],
            response_json["faces"],
            response_json["colors"],
            f"query: {text}",
        )
    else:
        print("Received error:", response.text)


if __name__ == "__main__":
    # test_text_query(text="couch")
    # test_text_query(text="things that should be regularly cleaned")
    # test_text_query(text="computer screen")
    # test_text_query(text="a good spot for a Christmas tree")

    # we used this in video demo
    test_text_query(text="things that might be dangerous to babies")
