"""3D OVS masks evaluated on their original pixel grid.

Render with the reconstruction's undistorted cameras, then sample predictions
at the undistorted location of each original pixel. No camera re-estimation.
"""
from pathlib import Path
import numpy as np
from PIL import Image
from tools.ovs_colmap import load_camera_data


def intrinsics(camera):
    p = camera.params
    if camera.model == 'PINHOLE':
        fx, fy, cx, cy = p
    elif camera.model in ('SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL'):
        fx, cx, cy = p[:3]
        fy = fx
    else:
        raise ValueError(f'Unsupported camera: {camera.model}')
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def decode_mask(path):
    with Image.open(path) as image:
        a = np.asarray(image.convert('RGBA'))
    # Ignore opaque alpha in ordinary RGB PNGs; transparent pixels are background.
    return ((a[..., :3].max(axis=-1) > 127) & (a[..., 3] > 127)).astype(np.uint8)


def load_frames(root, scene_name, prepared_root):
    root, prepared_root = Path(root), Path(prepared_root)
    raw = root / scene_name
    raw_cameras, raw_images = load_camera_data(raw / 'sparse/0')
    cameras, images = load_camera_data(prepared_root / 'sparse/0')
    raw_by_stem = {}
    for image in raw_images.values():
        stem = Path(image.name).stem
        if stem in raw_by_stem:
            raise ValueError(f'Ambiguous image stem: {stem}')
        raw_by_stem[stem] = image
    by_name = {image.name: image for image in images.values()}
    classes = [q.strip() for q in (raw / 'segmentations/classes.txt').read_text().splitlines() if q.strip()]
    if not classes or len(classes) != len(set(classes)):
        raise ValueError('Empty or duplicate classes')
    frames = []
    for folder in sorted((raw / 'segmentations').iterdir()):
        if not folder.is_dir():
            continue
        old_image = raw_by_stem[folder.name]
        image = by_name[old_image.name]
        old_camera, camera = raw_cameras[old_image.camera_id], cameras[image.camera_id]
        if camera.model not in ('PINHOLE', 'SIMPLE_PINHOLE'):
            raise ValueError('Prepared cameras must be undistorted pinhole cameras')
        if not (np.allclose(old_image.qvec2rotmat(), image.qvec2rotmat(), atol=1e-6)
                and np.allclose(old_image.tvec, image.tvec, atol=1e-6)):
            raise ValueError('Original and prepared camera poses differ')
        with Image.open(raw / 'images' / image.name) as rgb:
            raw_size = rgb.size
        with Image.open(prepared_root / 'images' / image.name) as rgb:
            prepared_size = rgb.size
        if raw_size != (old_camera.width, old_camera.height) or prepared_size != (camera.width, camera.height):
            raise ValueError('Camera/image resolution mismatch')
        mask_files = {p.stem.strip(): p for p in folder.glob('*.png')}
        if set(mask_files) != set(classes):
            raise ValueError(f'Mask/class mismatch in {folder}')
        masks = {q: decode_mask(mask_files[q]) for q in classes}
        if any(m.shape != raw_size[::-1] for m in masks.values()):
            raise ValueError(f'Mask/image resolution mismatch: {folder}')
        frames.append(dict(name=image.name, image=image, camera=camera,
                           raw_camera=old_camera, masks=masks, shape=raw_size[::-1],
                           render_shape=prepared_size[::-1]))
    if not frames:
        raise ValueError('No annotated frames found')
    return frames


def prediction_in_original_view(mask, frame):
    import cv2
    camera = frame['raw_camera']
    k = intrinsics(camera)
    distortion = np.zeros(5)
    if camera.model in ('SIMPLE_RADIAL', 'RADIAL'):
        distortion[0] = camera.params[3]
        if camera.model == 'RADIAL':
            distortion[1] = camera.params[4]
    h, w = frame['shape']
    result = np.zeros((h, w), dtype=mask.dtype)
    # Bound coordinate-map memory for original 4K images.
    for start in range(0, h, 128):
        yy, xx = np.mgrid[start:min(start + 128, h), :w]
        pixels = np.stack([xx, yy], axis=-1).astype(np.float32)
        mapped = cv2.undistortPoints(pixels.reshape(-1, 1, 2), k, distortion,
                                     P=intrinsics(frame['camera'])).reshape(pixels.shape)
        result[start:start + len(yy)] = cv2.remap(
            mask, mapped[..., 0], mapped[..., 1], cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return result
