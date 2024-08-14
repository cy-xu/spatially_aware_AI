from flask import Flask, jsonify, request
import json
import time

# from learning_utils_unity import InSituAdHocLearner
from clip_seem_fusion import InSituManager
from handy_utils import get_path, merge_objects, mark_object_of_interest

app = Flask(__name__)
# scan_name="Radha_desk",
# scan_name="5110_my_corner",
# scan_name="5110_foureyes",
# scan_name="seat_switch",

manager = InSituManager(
    scan_dir="scenes/iphone_3dscanner",
    scan_name="demo_scene",
    dataset="iphone",
    clip_patch_size=256,
    clip_patch_stride=128,
    voxel_size=0.04,
    trunc_vox=3,
    curr_ver=0,
)


@app.route("/reprocess_scan", methods=["POST"])
def reprocess_scan():
    # logging code for performance evaluation
    start_time = time.time()

    version = int(request.form["version"])
    print(f"received reprocess request, version: {version}")

    # update manager scene version
    manager.update_config(version)

    manager.run_clipfusion(
        scan_dir=manager.config.scene_inputdir,
        config=manager.config,
        device=manager.config.device,
        views_limit=0,
        curr_ver=version,
    )

    manager.save_files_and_broadcast(new_scene=True)
    
    # logging
    print(f"Processing time: {time.time()-start_time}")

    return jsonify({"message": "success"}), 200


@app.route("/text_query", methods=["POST"])
def text_query():
    # logging code for performance evaluation
    start_time = time.time()

    text_query = request.form["text_query"]
    print(f"received text query: {text_query}")

    if text_query.startswith("show me "):
        query = text_query[8:]
    else:
        query = text_query

    # CLIP query
    mesh_json = manager.clip_text_query(query)

    # logging
    print(f"Text query time: {time.time()-start_time}")

    if mesh_json is None:
        # return failure
        return jsonify({"error": "no object found"}), 404
    else:
        return jsonify(mesh_json), 200


@app.route("/requset_scene_mesh", methods=["POST"])
def requset_scene_mesh():
    scan_version = int(request.form["scan_version"])
    obj_key = request.form["obj_key"]
    mesh_type = request.form["color"]
    print(f"requested for a {mesh_type} mesh, ver.{scan_version}, obj_key: {obj_key}")

    # load the mesh or extract the object
    mesh_dict = manager.request_mesh(scan_version, obj_key, mesh_type)

    if mesh_dict is None:
        # return failure
        return jsonify({"error": "no object found"}), 404
    else:
        return jsonify(mesh_dict), 200


@app.route("/requset_scene_knowledge", methods=["POST"])
def requset_scene_knowledge():
    # load the scene knowledge json file for the current version
    scan_version = int(request.form["scan_version"])
    print(f"received request for scene knowledge, version: {scan_version}")

    json_path = get_path(manager.config, scan_version, "scene_knowledge")
    scene_knowledge = json.load(open(json_path, "r"))

    # update manager scene version
    manager.update_config(scan_version)

    # return jsonify(manager.scene_knowledge)
    return jsonify(scene_knowledge)


@app.route("/merge_objects", methods=["POST"])
def client_merge_objects():
    # the client sends a list of objects keys to merge as one
    object_list = request.form["object_list"]
    object_list = json.loads(object_list)["items"]
    new_name = request.form["new_name"]
    print(f"receveid merge objects: {object_list}, new name: {new_name}")

    # modify the scene knowledge json file
    new_name, manager.scene_knowledge = merge_objects(
        manager.scene_knowledge,
        manager.vertex_obj_idx,
        manager.insitu_model,
        object_list,
        new_name,
    )
    # save the changes to disk
    manager.save_files_and_broadcast(new_scene=False)
    return new_name, 200


@app.route("/rename_object", methods=["POST"])
def client_rename_object():
    # the client sends one object key and its new name
    object_key = request.form["object_key"]
    new_name = request.form["new_name"]
    print(f"receveid rename object {object_key} to {new_name}")

    # modify the scene knowledge json file
    new_name, manager.scene_knowledge = merge_objects(
        manager.scene_knowledge,
        manager.vertex_obj_idx,
        manager.insitu_model,
        [object_key],
        new_name,
    )
    # save the changes to disk
    manager.save_files_and_broadcast(new_scene=False)
    return new_name, 200


@app.route("/memorize_objects", methods=["POST"])
def memorize_objects():
    # the client sends one object key and its new name
    object_list = request.form["object_list"]
    object_list = json.loads(object_list)["items"]
    print(f"receveid memorize objects: {object_list}")

    manager.scene_knowledge = mark_object_of_interest(
        manager.scene_knowledge, manager.insitu_model, object_list
    )
    # save the changes to disk
    manager.save_files_and_broadcast(new_scene=False)
    return jsonify({"message": "success"}), 200


@app.route("/insitu_learn", methods=["GET"])
def insitu_learn():
    print(
        f"received insitu learn request, train model and reprocess CURRENT scan version:{manager.curr_ver}"
    )
    # when client calls this, we save all changes and train the model
    manager.save_files_and_broadcast(new_scene=False)

    # train model
    manager.insitu_model.prepare_data(manager)
    manager.insitu_model.train_model()

    return jsonify({"message": "success"}), 200


@app.route("/copy_object", methods=["POST"])
def copy_object():
    obj_id = request.form["obj_id"]
    mesn_json = manager.unity_copy_object(obj_id)
    return jsonify(mesn_json), 200


@app.route("/reset_insitu_model", methods=["POST"])
def reset_insitu_model():
    manager.insitu_model.reset_model(delete_weights=True)
    return jsonify({"message": "success"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3291)
