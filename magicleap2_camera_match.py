import os
import cv2
import json
import numpy as np
import OpenEXR
import Imath

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def readEXR_depth(filename):
    """Read color + depth data from EXR image file."""
    # Open the input file
    depth_exr = OpenEXR.InputFile(filename)

    # Compute size
    dw = depth_exr.header()["dataWindow"]
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Read the three color channels as 32-bit floats
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    depth = np.frombuffer(depth_exr.channel("Y", FLOAT), dtype=np.float32)
    depth.shape = (size[1], size[0])  # reshape for 2D image

    return depth


# resize depth image to match rgb image


def get_intrinsics(meta):
    width = meta["intrinsics"]["Width"]
    height = meta["intrinsics"]["Height"]
    fl_x = meta["intrinsics"]["FocalLength"]["x"]
    fl_y = meta["intrinsics"]["FocalLength"]["y"]
    pp_x = meta["intrinsics"]["PrincipalPoint"]["x"]
    pp_y = meta["intrinsics"]["PrincipalPoint"]["y"]

    camera_matrix = np.array([[fl_x, 0, pp_x], [0, fl_y, pp_y], [0, 0, 1]])
    distortion_coeffs = np.array(meta["intrinsics"]["Distortion"])
    return camera_matrix, distortion_coeffs


def get_extrinsic(meta):
    pose = meta["pose"]
    extrinsic_matrix = np.array(
        [
            [pose["e00"], pose["e01"], pose["e02"], pose["e03"]],
            [pose["e10"], pose["e11"], pose["e12"], pose["e13"]],
            [pose["e20"], pose["e21"], pose["e22"], pose["e23"]],
            [pose["e30"], pose["e31"], pose["e32"], pose["e33"]],
        ]
    )
    return extrinsic_matrix


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    elif width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


base_dir = "scenes/magicleap2/1231_1218"
rgb_images = sorted(os.listdir(base_dir + "/rgb"))
depth_images = sorted(os.listdir(base_dir + "/depth"))
rgb_poses = sorted(os.listdir(base_dir + "/rgbpose"))
depth_poses = sorted(os.listdir(base_dir + "/depthpose"))
os.makedirs(base_dir + "/depth_undistorted", exist_ok=True)
os.makedirs(base_dir + "/rgb_undistorted", exist_ok=True)
os.makedirs(base_dir + "/rgb_registered", exist_ok=True)
os.makedirs(base_dir + "/depth_registered", exist_ok=True)

assert len(rgb_images) == len(depth_images) == len(rgb_poses) == len(depth_poses)

for i in range(len(rgb_images)):
    rgb_img = cv2.imread(base_dir + "/rgb/" + rgb_images[i])
    depth_img = readEXR_depth(base_dir + "/depth/" + depth_images[i])

    # read the meta files
    with open(base_dir + "/rgbpose/" + rgb_poses[i]) as f:
        rgb_meta = json.load(f)
    with open(base_dir + "/depthpose/" + depth_poses[i]) as f:
        depth_meta = json.load(f)

    rgb_intrinsics, rgb_dist = get_intrinsics(rgb_meta)
    depth_intrinsics, depth_dist = get_intrinsics(depth_meta)

    rgb_extrinsic = get_extrinsic(rgb_meta)
    depth_extrinsic = get_extrinsic(depth_meta)

    # # Compute the aspect ratio resize
    # depth_img = resize_with_aspect_ratio(depth_img, width=rgb_img.shape[1])

    # # depth_img is greater than rgb_img in terms of height, crop it
    # # if depth_img.shape[0] > rgb_img.shape[0]:
    # #     start_y = (depth_img.shape[0] - rgb_img.shape[0]) // 2
    # #     depth_img = depth_img[start_y : start_y + rgb_img.shape[0], :]

    # # adjust the intrinsics
    # scale_x = rgb_img.shape[1] / depth_img.shape[1]
    # scale_y = rgb_img.shape[0] / depth_img.shape[0]
    # depth_intrinsics[0] *= scale_x
    # depth_intrinsics[1] *= scale_y

    # undistort the images
    rgb_img = cv2.undistort(rgb_img, rgb_intrinsics, rgb_dist)
    depth_img = cv2.undistort(depth_img, depth_intrinsics, depth_dist)

    # colorize the depth image for visualization
    depth_img_viz = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
    depth_img_viz = cv2.applyColorMap(
        (depth_img_viz * 255).astype(np.uint8), cv2.COLORMAP_JET
    )

    # save the colorized depth image
    cv2.imwrite(base_dir + f"/depth_undistorted/depth_color_{i+1}.png", depth_img_viz)
    cv2.imwrite(base_dir + f"/rgb_undistorted/rgb_undistorted_{i+1}.png", rgb_img)
    print(f"colorized depth {i+1} saved")

    # # undistort the depth image
    # cv2.imwrite("undistorted_rgb.png", rgb_img)
    # # normalize the depth image to [0, 1] before saving
    # depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
    # cv2.imwrite("undistorted_depth.png", depth_img)

    # Compute the relative pose of the depth camera with respect to the RGB camera
    R_depth_to_rgb = np.linalg.inv(depth_extrinsic[:3, :3]) @ rgb_extrinsic[:3, :3]
    t_depth_to_rgb = rgb_extrinsic[:3, 3] - R_depth_to_rgb @ depth_extrinsic[:3, 3]

    # Initialize depth-to-RGB mapping
    depth_to_rgb = np.zeros((depth_img.shape[0], depth_img.shape[1], 2))

    h, w = depth_img.shape[:2]

    # Compute depth-to-RGB mapping
    for v in range(h):
        for u in range(w):
            # Unproject depth to 3D
            X = np.linalg.inv(depth_intrinsics) @ [
                u * depth_img[v, u],
                v * depth_img[v, u],
                depth_img[v, u],
            ]

            # Add extra dimension for homogeneous coordinates
            X = np.append(X, 1)
            # Transform to RGB camera's frame
            X_trans = R_depth_to_rgb @ X[:3] + t_depth_to_rgb

            # Normalize to remove scale
            X_trans /= X_trans[2]
            u_rgb, v_rgb, _ = rgb_intrinsics @ X_trans
            depth_to_rgb[v, u] = [u_rgb, v_rgb]

    # Interpolate RGB values at depth image's pixel locations
    registered_rgb_img = cv2.remap(
        rgb_img, depth_to_rgb.astype(np.float32), None, cv2.INTER_LINEAR
    )

    # Update intrinsic parameters for cropping
    rgb_intrinsics_corrected, roi = cv2.getOptimalNewCameraMatrix(
        rgb_intrinsics,
        rgb_dist,
        (rgb_img.shape[1], rgb_img.shape[0]),
        1,
        (rgb_img.shape[1], rgb_img.shape[0]),
    )

    # Crop both images
    x, y, w, h = roi
    # registered_rgb_img = registered_rgb_img[y : y + h, x : x + w]
    # depth_cropped = depth_img[y : y + h, x : x + w]

    # Save the transformed and cropped images
    cv2.imwrite(
        base_dir + f"/rgb_registered/rgb_registered_{i+1}.png", registered_rgb_img
    )
    cv2.imwrite(base_dir + f"/depth_registered/depth_registered_{i+1}.png", depth_img)

    breakpoint()
