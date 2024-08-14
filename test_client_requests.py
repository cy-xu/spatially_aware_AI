import numpy as np
import requests
import json
import open3d as o3d
import pretty_errors

server = "http://127.0.0.1:3291/"
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


def reprocess_scan(version=0):
    """
    Test reprocessing a scene
    """
    data = {"version": version}
    response = requests.post(server + "reprocess_scan", data=data)
    response_json = json.loads(response.text)


def switch_context(target_version):
    """
    Test requesting scene knowledge / set context
    """
    # switching context to v00
    data = {"scan_version": target_version}
    response = requests.post(server + "requset_scene_knowledge", data=data)
    scene_knowledge = json.loads(response.text)

    return scene_knowledge


def test_rename_objects_and_train():
    """
    Test labeling objects in v00, tarin model, and reprocess v01
    """

    if False:
        print(f"mimic user input, merge floor and rug")
        floor_list = ["floor:1", "rug:1", "floor:21"]
        response = requests.post(
            server + "merge_objects",
            data={
                "object_list": json.dumps({"items": floor_list}),
                "new_name": "merged_floor",
            },
        )

        wall_list = ["wall:2", "wall-tile:1"]
        response = requests.post(
            server + "merge_objects",
            data={
                "object_list": json.dumps({"items": wall_list}),
                "new_name": "merged_wall",
            },
        )

        print(f"mimic user input, rename cardbord:7 and check if it's missing")
        response = requests.post(
            server + "rename_object",
            data={"object_key": "cardboard:7", "new_name": "Missing_box"},
        )
        response = requests.post(
            server + "rename_object",
            data={"object_key": "chair:1", "new_name": "Missing_chair"},
        )

        response = requests.post(
            server + "rename_object",
            data={"object_key": "couch:1", "new_name": "Unchanged_couch"},
        )

        response = requests.post(
            server + "rename_object",
            data={"object_key": "tv:2", "new_name": "Unchanged_screen"},
        )

    # train model, send a get request to /insitu_learn
    response = requests.get(server + "insitu_learn")
    print(f"model training done, response: {response.text}")

    # # reprocess v00
    # # response = requests.post(server + "reprocess_scan", data={"version": 0})

    # switch to v01 and reprocess
    response = requests.post(server + "reprocess_scan", data={"version": 1})


def test_requesting_meshes(version=1, color="rgb", obj_key="scene"):
    """
    Test requesting scene / object mesh
    """
    if obj_key in ["missing", "unchanged"]:
        # switch context to v01
        scene_knowledge = switch_context(target_version=1)

    data = {"scan_version": version, "obj_key": obj_key, "color": color}

    response = requests.post(server + "requset_scene_mesh", data=data)
    print(f"requesting mesh {color} for {obj_key} in version {version}")

    if response.ok:
        print(response.status_code)
        response_json = json.loads(response.text)
    else:
        print("Received error:", response.text)

    if "faces" in response_json.keys():
        draw_a_mesh(
            response_json["vertices"],
            response_json["faces"],
            response_json["colors"],
            f"scene_{version}_{color}",
        )
    else:
        for obj in response_json.keys():
            print(f'saving mesh for object "{obj}')
            draw_a_mesh(
                response_json[obj]["vertices"],
                response_json[obj]["faces"],
                response_json[obj]["colors"],
                f"{obj}_{version}_{color}",
            )


if __name__ == "__main__":
    """
    reset model sequence
    1. stop the Flask server
    2. delete output directory
    3. restart the Flask server, v00 reproceses automatically
    """

    # reprocess_scan(version=0)
    # switch_context(target_version=1)
    # test_requesting_meshes(version=1, color="rgb", obj_key="unchanged")
    # test_requesting_meshes(version=0, color="rgb", obj_key="all_objects")

    # test_text_query(text="couch")
    # test_requesting_meshes(version=0, color="segmentation", obj_key="scene")
    # test_text_query(text="things that should be regularly cleaned")
    test_text_query(text="things that might be dangerous to babies")
    # test_text_query(text="computer screen")
    # test_text_query(text="a good spot for a Christmas tree")

    # reset model first
    # response = requests.get(server + "reset_insitu_model")
    # test_rename_objects_and_train()
