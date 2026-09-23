"""COLMAP binary readers reused from SceneSplat LERF evaluation."""
import collections
import struct
import os
import numpy as np

CameraModel = collections.namedtuple(
    "CameraModel",
    ["model_id", "model_name", "num_params"]
)

Camera = collections.namedtuple(
    "Camera",
    ["id", "model", "width", "height", "params"]
)

BaseImage = collections.namedtuple(
    "Image",
    [
        "id",
        "qvec",
        "tvec",
        "camera_id",
        "name",
        "xys",
        "point3D_ids",
    ],
)

CAMERA_MODELS = {
    CameraModel(0, "SIMPLE_PINHOLE", 3),
    CameraModel(1, "PINHOLE", 4),
    CameraModel(2, "SIMPLE_RADIAL", 4),
    CameraModel(3, "RADIAL", 5),
    CameraModel(4, "OPENCV", 8),
    CameraModel(5, "OPENCV_FISHEYE", 8),
    CameraModel(6, "FULL_OPENCV", 12),
    CameraModel(7, "FOV", 5),
    CameraModel(8, "SIMPLE_RADIAL_FISHEYE", 4),
    CameraModel(9, "RADIAL_FISHEYE", 5),
    CameraModel(10, "THIN_PRISM_FISHEYE", 12),
}

CAMERA_MODEL_IDS = {
    model.model_id: model
    for model in CAMERA_MODELS
}
# Quaternion â†’ Rotation Matrix

def qvec2rotmat(qvec):

    return np.array([
        [
            1 - 2*qvec[2]**2 - 2*qvec[3]**2,
            2*qvec[1]*qvec[2] - 2*qvec[0]*qvec[3],
            2*qvec[3]*qvec[1] + 2*qvec[0]*qvec[2],
        ],
        [
            2*qvec[1]*qvec[2] + 2*qvec[0]*qvec[3],
            1 - 2*qvec[1]**2 - 2*qvec[3]**2,
            2*qvec[2]*qvec[3] - 2*qvec[0]*qvec[1],
        ],
        [
            2*qvec[3]*qvec[1] - 2*qvec[0]*qvec[2],
            2*qvec[2]*qvec[3] + 2*qvec[0]*qvec[1],
            1 - 2*qvec[1]**2 - 2*qvec[2]**2,
        ],
    ])


class Image(BaseImage):

    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


# Read Binary Bytes

def read_next_bytes(
    fid,
    num_bytes,
    format_char_sequence,
    endian_character="<",
):

    data = fid.read(num_bytes)

    return struct.unpack(
        endian_character + format_char_sequence,
        data,
    )

# Read COLMAP Cameras

def read_intrinsics_binary(path_to_model_file):

    cameras = {}

    with open(path_to_model_file, "rb") as fid:

        num_cameras = read_next_bytes(
            fid,
            8,
            "Q",
        )[0]

        for _ in range(num_cameras):

            camera_properties = read_next_bytes(
                fid,
                24,
                "iiQQ",
            )

            camera_id = camera_properties[0]

            model_id = camera_properties[1]

            model_name = CAMERA_MODEL_IDS[
                model_id
            ].model_name

            width = camera_properties[2]
            height = camera_properties[3]

            num_params = CAMERA_MODEL_IDS[
                model_id
            ].num_params

            params = read_next_bytes(
                fid,
                8 * num_params,
                "d" * num_params,
            )

            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )

    return cameras

# Read COLMAP Images

def read_extrinsics_binary(path_to_model_file):

    images = {}

    with open(path_to_model_file, "rb") as fid:

        num_images = read_next_bytes(
            fid,
            8,
            "Q",
        )[0]

        for _ in range(num_images):

            binary = read_next_bytes(
                fid,
                64,
                "idddddddi",
            )

            image_id = binary[0]

            qvec = np.array(binary[1:5])

            tvec = np.array(binary[5:8])

            camera_id = binary[8]

            image_name = ""

            current_char = read_next_bytes(
                fid,
                1,
                "c",
            )[0]

            while current_char != b"\x00":

                image_name += current_char.decode("utf-8")

                current_char = read_next_bytes(
                    fid,
                    1,
                    "c",
                )[0]

            num_points2D = read_next_bytes(
                fid,
                8,
                "Q",
            )[0]

            x_y_id_s = read_next_bytes(
                fid,
                24 * num_points2D,
                "ddq" * num_points2D,
            )

            xys = np.column_stack([
                tuple(map(float, x_y_id_s[0::3])),
                tuple(map(float, x_y_id_s[1::3])),
            ])

            point3D_ids = np.array(
                tuple(map(int, x_y_id_s[2::3]))
            )

            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )

    return images


# Load COLMAP Camera Data

def load_camera_data(colmap_sparse_folder):

    print("\n[INFO] Loading COLMAP camera data...\n")

    cameras = read_intrinsics_binary(
        os.path.join(
            colmap_sparse_folder,
            "cameras.bin",
        )
    )

    images = read_extrinsics_binary(
        os.path.join(
            colmap_sparse_folder,
            "images.bin",
        )
    )

    print(f"Loaded cameras : {len(cameras)}")
    print(f"Loaded images  : {len(images)}\n")

    return cameras, images



