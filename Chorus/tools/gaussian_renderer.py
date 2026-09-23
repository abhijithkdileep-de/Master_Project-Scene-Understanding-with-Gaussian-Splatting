# Adapted from SceneSplat/tools/gaussian_renderer.py; selected-splat footprint renderer.
import numpy as np


# Quaternion utilities

def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion [w, x, y, z] to a 3x3 rotation matrix.

    Parameters
    ----------
    q : array-like, shape (4,)

    Returns
    -------
    R : ndarray, shape (3, 3)
    """

    q = np.asarray(q, dtype=np.float64)

    norm = np.linalg.norm(q)

    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)

    q = q / norm

    w, x, y, z = q

    return np.array(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ],
            [
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
            ],
            [
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


# Camera utilities

def get_camera_intrinsics(camera):
    """
    Return fx, fy, cx, cy for common COLMAP camera models.

    For the current teatime scene, camera.model == PINHOLE.
    """

    p = np.asarray(camera.params, dtype=np.float64)

    if camera.model == "PINHOLE":

        fx, fy, cx, cy = p[:4]

    elif camera.model == "SIMPLE_PINHOLE":

        f, cx, cy = p[:3]

        fx = f
        fy = f

    elif camera.model == "SIMPLE_RADIAL":

        # Use pinhole intrinsics; the covariance projection does not model lens distortion.
        f, cx, cy = p[:3]

        fx = f
        fy = f

    elif camera.model == "RADIAL":

        f, cx, cy = p[:3]

        fx = f
        fy = f

    elif camera.model in (
        "OPENCV",
        "FULL_OPENCV",
        "OPENCV_FISHEYE",
    ):

        fx, fy, cx, cy = p[:4]

    else:

        raise RuntimeError(
            f"Unsupported camera model for Gaussian renderer: "
            f"{camera.model}"
        )

    return float(fx), float(fy), float(cx), float(cy)


# Opacity utility

def normalize_opacity(opacity):
    """
    Convert opacity values into [0, 1].

    If the input already looks like probabilities, leave it alone.
    Otherwise treat values as logits and apply sigmoid.
    """

    opacity = np.asarray(opacity, dtype=np.float64)

    if opacity.size == 0:
        return opacity

    if opacity.min() >= 0.0 and opacity.max() <= 1.0:

        return opacity

    opacity = 1.0 / (1.0 + np.exp(-opacity))

    return opacity


# 3D covariance

def build_world_covariance(scale, quat):
    """
    Construct the 3D Gaussian covariance:

        Sigma_world = R diag(sx^2, sy^2, sz^2) R^T
    """

    scale = np.asarray(scale, dtype=np.float64)

    # Prevent exactly-zero covariance
    scale = np.maximum(scale, 1e-8)

    Rg = quaternion_to_rotation_matrix(quat)

    S = np.diag(scale * scale)

    covariance = Rg @ S @ Rg.T

    return covariance


# Project one covariance into image space

def project_covariance_to_image(
    xyz_cam,
    covariance_cam,
    fx,
    fy,
):
    """
    First-order projection of a 3D covariance into image space.

    Projection:
        u = fx * X/Z + cx
        v = fy * Y/Z + cy

    J is the Jacobian of [u,v] w.r.t [X,Y,Z].

        Sigma_2d = J Sigma_cam J^T
    """

    X, Y, Z = xyz_cam

    if Z <= 1e-8:
        return None

    J = np.array(
        [
            [
                fx / Z,
                0.0,
                -fx * X / (Z * Z),
            ],
            [
                0.0,
                fy / Z,
                -fy * Y / (Z * Z),
            ],
        ],
        dtype=np.float64,
    )

    covariance_2d = (
        J
        @ covariance_cam
        @ J.T
    )

    # Numerical regularization
    covariance_2d += np.eye(2) * 1e-6

    return covariance_2d


# Rasterize one 2D Gaussian

def rasterize_gaussian_patch(
    soft_mask,
    center_x,
    center_y,
    covariance_2d,
    opacity,
    sigma_extent=3.0,
    max_radius=60,
):
    """
    Rasterize an anisotropic Gaussian into soft_mask.

    Accumulation uses probabilistic union:

        M <- 1 - (1-M)(1-alpha)

    This produces smooth overlapping Gaussian regions.
    """

    H, W = soft_mask.shape

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance_2d
    )

    eigenvalues = np.maximum(
        eigenvalues,
        1e-8
    )

    # Largest projected sigma determines bounding box
    max_sigma = np.sqrt(
        eigenvalues.max()
    )

    radius = int(
        np.ceil(
            sigma_extent * max_sigma
        )
    )

    radius = int(
        np.clip(
            radius,
            1,
            max_radius,
        )
    )

    x0 = max(
        int(np.floor(center_x)) - radius,
        0,
    )

    x1 = min(
        int(np.ceil(center_x)) + radius + 1,
        W,
    )

    y0 = max(
        int(np.floor(center_y)) - radius,
        0,
    )

    y1 = min(
        int(np.ceil(center_y)) + radius + 1,
        H,
    )

    if x0 >= x1 or y0 >= y1:
        return

    yy, xx = np.mgrid[
        y0:y1,
        x0:x1
    ]

    dx = xx - center_x
    dy = yy - center_y

    delta = np.stack(
        [dx, dy],
        axis=-1,
    )

    try:

        inverse_covariance = np.linalg.inv(
            covariance_2d
        )

    except np.linalg.LinAlgError:

        return

    mahalanobis = np.einsum(
        "...i,ij,...j->...",
        delta,
        inverse_covariance,
        delta,
    )

    gaussian = np.exp(
        -0.5 * mahalanobis
    )

    alpha = (
        opacity * gaussian
    )

    alpha = np.clip(
        alpha,
        0.0,
        0.999,
    )

    patch = soft_mask[
        y0:y1,
        x0:x1
    ]

    # Soft union of overlapping splats
    patch[:] = (
        1.0
        - (1.0 - patch)
        * (1.0 - alpha)
    )

# Gaussian visibility using renderer geometry

def compute_renderer_visibility(
    coord,
    opacity,
    image,
    camera,
    image_shape,
    max_radius=60,
    opacity_threshold=0.01,
):
    """
    Determine which SceneSplat Gaussians are renderable in
    a specific camera using the same visibility conditions
    as render_semantic_mask().

    A Gaussian is considered visible when:
      1. it is in front of the camera,
      2. its projected center lies within the renderer's
         screen margin,
      3. its opacity passes the renderer threshold.

    Returns
    -------
    visible : bool ndarray [N]
        Visibility mask for all Gaussians.
    """

    H, W = image_shape

    fx, fy, cx, cy = get_camera_intrinsics(
        camera
    )

    R_camera = image.qvec2rotmat()

    t_camera = np.asarray(
        image.tvec,
        dtype=np.float64,
    )

    coord = np.asarray(
        coord,
        dtype=np.float64,
    )

    opacity = normalize_opacity(
        opacity
    )

    # Use the same world-to-camera transform as render_semantic_mask().

    xyz_cam = (
        coord @ R_camera.T
        + t_camera
    )

    X = xyz_cam[:, 0]
    Y = xyz_cam[:, 1]
    Z = xyz_cam[:, 2]

    # In front of camera

    valid_depth = (
        Z > 1e-8
    )

    # Avoid invalid division for behind-camera Gaussians
    safe_Z = np.where(
        valid_depth,
        Z,
        1.0
    )

    # Use the same perspective projection as render_semantic_mask().

    u = (
        fx * X / safe_Z
        + cx
    )

    v = (
        fy * Y / safe_Z
        + cy
    )

    # Match the renderer bounds, including the max_radius margin.

    valid_screen = (
        (u >= -max_radius)
        & (u < W + max_radius)
        & (v >= -max_radius)
        & (v < H + max_radius)
    )

    # Same opacity test as renderer

    valid_opacity = (
        opacity >= opacity_threshold
    )

    visible = (
        valid_depth
        & valid_screen
        & valid_opacity
    )

    return visible

# Main SceneSplat semantic renderer

def render_semantic_mask(
    coord,
    scale,
    quat,
    opacity,
    selected_indices,
    image,
    camera,
    image_shape,
    mask_threshold=0.05,
    sigma_extent=3.0,
    max_radius=60,
    opacity_threshold=0.01,
):
    """
    Render selected SceneSplat Gaussians into a dense 2D
    semantic mask.

    Parameters
    ----------
    coord:
        Gaussian centers [N,3]

    scale:
        Gaussian anisotropic scales [N,3]

    quat:
        Gaussian quaternion [N,4]

    opacity:
        Gaussian opacity [N]

    selected_indices:
        indices selected by the SceneSplat semantic threshold

    image:
        COLMAP Image object with qvec/tvec

    camera:
        COLMAP Camera object

    image_shape:
        (H, W)

    Returns
    -------
    binary_mask : uint8 [H,W]
    soft_mask   : float32 [H,W]
    """

    H, W = image_shape

    fx, fy, cx, cy = get_camera_intrinsics(
        camera
    )

    # Camera transform

    R_camera = image.qvec2rotmat()

    t_camera = np.asarray(
        image.tvec,
        dtype=np.float64,
    )

    coord = np.asarray(
        coord,
        dtype=np.float64,
    )

    scale = np.asarray(
        scale,
        dtype=np.float64,
    )

    quat = np.asarray(
        quat,
        dtype=np.float64,
    )

    opacity = normalize_opacity(
        opacity
    )

    selected_indices = np.asarray(
        selected_indices,
        dtype=np.int64,
    )

    soft_mask = np.zeros(
        (H, W),
        dtype=np.float32,
    )

    rendered = 0
    skipped_depth = 0
    skipped_screen = 0
    skipped_opacity = 0

    # Render selected semantic Gaussians

    for idx in selected_indices:

        xyz_world = coord[idx]

        xyz_cam = (
            R_camera @ xyz_world
            + t_camera
        )

        X, Y, Z = xyz_cam

        if Z <= 1e-8:

            skipped_depth += 1

            continue

        u = (
            fx * X / Z
            + cx
        )

        v = (
            fy * Y / Z
            + cy
        )

        if (
            u < -max_radius
            or u >= W + max_radius
            or v < -max_radius
            or v >= H + max_radius
        ):

            skipped_screen += 1

            continue

        alpha = float(
            opacity[idx]
        )

        if alpha < opacity_threshold:

            skipped_opacity += 1

            continue

        # 3D Gaussian covariance

        covariance_world = build_world_covariance(
            scale[idx],
            quat[idx],
        )

        # Transform covariance world -> camera
        covariance_camera = (
            R_camera
            @ covariance_world
            @ R_camera.T
        )

        # Project covariance into image

        covariance_2d = project_covariance_to_image(
            xyz_cam,
            covariance_camera,
            fx,
            fy,
        )

        if covariance_2d is None:

            continue

        # Rasterize

        rasterize_gaussian_patch(
            soft_mask=soft_mask,
            center_x=u,
            center_y=v,
            covariance_2d=covariance_2d,
            opacity=alpha,
            sigma_extent=sigma_extent,
            max_radius=max_radius,
        )

        rendered += 1

    binary_mask = (
        soft_mask >= mask_threshold
    ).astype(np.uint8)

    print(
        "\n[Gaussian Renderer]"
    )

    print(
        f"Selected Gaussians : "
        f"{len(selected_indices)}"
    )

    print(
        f"Rendered Gaussians : "
        f"{rendered}"
    )

    print(
        f"Skipped depth      : "
        f"{skipped_depth}"
    )

    print(
        f"Skipped screen     : "
        f"{skipped_screen}"
    )

    print(
        f"Skipped opacity    : "
        f"{skipped_opacity}"
    )

    print(
        f"Soft-mask range    : "
        f"{soft_mask.min():.4f} "
        f"to "
        f"{soft_mask.max():.4f}"
    )

    print(
        f"Mask pixels        : "
        f"{binary_mask.sum()}"
    )

    return (
        binary_mask,
        soft_mask,
    )
