import os
import shutil
import sys

import torch
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image


def _prepend_env_path(var_name, paths):
    existing = os.environ.get(var_name, "")
    parts = [path for path in existing.split(os.pathsep) if path]
    for path in reversed(paths):
        if path and os.path.isdir(path) and path not in parts:
            parts.insert(0, path)
    if parts:
        os.environ[var_name] = os.pathsep.join(parts)


def _prepare_gsplat_jit_env():
    env_root = sys.prefix
    include_dirs = [
        os.path.join(env_root, "targets", "x86_64-linux", "include"),
        os.path.join(
            env_root,
            "lib",
            f"python{sys.version_info.major}.{sys.version_info.minor}",
            "site-packages",
            "nvidia",
            "cuda_runtime",
            "include",
        ),
    ]
    _prepend_env_path("CPATH", include_dirs)
    _prepend_env_path("CPLUS_INCLUDE_PATH", include_dirs)

    env_bin = os.path.join(env_root, "bin")
    if shutil.which("nvcc") is None and os.path.exists(os.path.join(env_bin, "nvcc")):
        _prepend_env_path("PATH", [env_bin])
    os.environ.setdefault("CUDA_HOME", env_root)
    os.environ.setdefault("CUDA_PATH", env_root)


def _rasterization(**kwargs):
    _prepare_gsplat_jit_env()
    try:
        import gsplat
    except ImportError as exc:
        raise ImportError(
            "2D adaptation rendering requires gsplat==1.4.0. "
            "Install it in the active environment before running 2D adaptation."
        ) from exc
    return gsplat.rasterization(**kwargs)


def build_gs_dict(coord, feat, offset):
    if feat.shape[-1] == 11:
        # no coord
        color = feat[:, 0:3]
        opacity = feat[:, 3:4][:, 0]  # opacity
        quat = feat[:, 4:8]
        scale = feat[:, 8:11]
        #  ("color", "opacity", "quat", "scale")
    else: 
        raise NotImplementedError("feat channels not supported:", feat.shape[-1])
    
    color = (color + 1) * 127.5  / 255.0  # color, -1~1 to 0~1
    gs_dict_list = []

    gs_dict_batch ={
        "means": coord,
        "color": color,
        "opacity": opacity,
        "quat": quat,
        "scale": scale,
        }

    offset = torch.cat((torch.tensor([0], device=offset.device), offset))
    for j in range(len(offset) - 1):
        start, end = offset[j], offset[j + 1]
        gs_dict_j = {}
        for k, v in gs_dict_batch.items():
            gs_dict_j[k] = v[start:end]
        
        gs_dict_list.append(gs_dict_j)
    
    return gs_dict_list
    # here for multi batch data, we convert to list for easy rendering later





def rasterize_multiple_gaussians_to_multiple_feats(
        gs_dict,
        viewmats, # (N, 4, 4)
        Ks, # (N, 3, 3)
        width,
        height,
        features_3d_list,  #  B, N, C
        save_visualize=False,
        need_grad=False,
        downsample_ratio=16,
        image_paths_batch=None,
        epoch_progress=0,
        ):

    feats_list = []
    valid_feats_list = []

    for j, gs_dict_j in enumerate(gs_dict):
        viewmat_j = viewmats[j] # S,4,4
        Kj = Ks[j] # S, 3,3
        features_3d_batch_j = features_3d_list[j]  # N, C
        for view_i in range(viewmat_j.shape[0]):
            viewmat_i = viewmat_j[view_i:view_i+1]
            K_i = Kj[view_i:view_i+1].clone()

            # downsample K, width and height
            K_i[:, 0, 0] /= downsample_ratio
            K_i[:, 1, 1] /= downsample_ratio
            K_i[:, 0, 2] /= downsample_ratio
            K_i[:, 1, 2] /= downsample_ratio
            width_j = width // downsample_ratio
            height_j = height // downsample_ratio

            if not need_grad:
                with torch.no_grad():
                    feats, alphas, meta = _rasterization(
                        means = gs_dict_j['means'].float(),
                        quats = gs_dict_j['quat'].float(),
                        scales = gs_dict_j['scale'].float(),
                        colors = features_3d_batch_j.float(), # 0 - 1
                        opacities = gs_dict_j['opacity'].float(), # after sigmoid
                        viewmats = viewmat_i.float(),
                        Ks = K_i.float(),
                        width = width_j ,
                        height = height_j ,
                    )
            else:
                feats, alphas, meta = _rasterization(
                    means = gs_dict_j['means'].float(),
                    quats = gs_dict_j['quat'].float(),
                    scales = gs_dict_j['scale'].float(),
                    colors = features_3d_batch_j.float(), # 0 - 1
                    opacities = gs_dict_j['opacity'].float(), # after sigmoid
                    viewmats = viewmat_i.float(),
                    Ks = K_i.float(),
                    width = width_j ,
                    height = height_j ,
                )
                    
            valid_feats = alphas > 1e-6
            valid_feats_list.append(valid_feats)
            feats_list.append(feats)
            
    if save_visualize:
        j = 0
        view_i = 0
        images_name = image_paths_batch[j][view_i].split("/")[-1]
        # os.path.join(save_dir, f"Epoch_{int(epoch_progress*1000)}_DINO2DFeat.png")
        target_save_name = os.path.join("./debug_dino_vis", f"Epoch_{int(epoch_progress*1000)}_DINO2DFeat_render_{images_name}.png")
        if not os.path.exists("./debug_dino_vis"):
            os.makedirs("./debug_dino_vis")
        if not os.path.exists(target_save_name):
            # images_name = image_paths_batch[j][view_i].split("/")[-1]
            # do pca on render feats
            
            pca = PCA(n_components=3)
            feats_np = feats_list[0][0].detach().cpu().numpy()
            # feats[0].detach().cpu().numpy()
            h, w, c = feats_np.shape
            feats_np_2d = feats_np.reshape(-1, c)
            feats_pca = pca.fit_transform(feats_np_2d)
            feats_pca = (feats_pca - feats_pca.min()) / (feats_pca.max() - feats_pca.min())
            feats_pca = (feats_pca * 255).astype(np.uint8).reshape(h, w, 3)
            
            img = Image.fromarray(feats_pca)
            # epoch_progress_in_percent = int(epoch_progress * 100)
            # epoch_progress_in_percent = int(epoch_progress * 1000)
            img.save(target_save_name)
            print(f"Saved debug dino render feat image to {target_save_name}")
        # img.save(f"./debug_dino_vis/b{j}_view{view_i}_{images_name}_render_epoch{epoch_progress_in_percent}.png")
    
    feats_list = torch.cat(feats_list, dim=0)  # (N, H, W, C)
    valid_feats_list = torch.cat(valid_feats_list, dim=0)  # (N, H, W)

    return feats_list, valid_feats_list


def rasterize_multiple_gaussians_to_multiple_imgs(
        gs_dict,
        viewmats, # (N, 4, 4)
        Ks, # (N, 3, 3)
        width,
        height,
        image_paths_batch,
        save_visualize=False,
        target_render_save_name=None,
    ):
    # TODO, use offset to cut gaussians between cenes


    colors_list = []

    for j, gs_dict_j in enumerate(gs_dict):
        viewmat_j = viewmats[j] # S,4,4
        Kj = Ks[j] # S, 3,3
        
        for view_i in range(viewmat_j.shape[0]):
            viewmat_i = viewmat_j[view_i:view_i+1]
            K_i = Kj[view_i:view_i+1].clone()
            with torch.no_grad():
                colors, alphas, meta = _rasterization(
                    means = gs_dict_j['means'].float(),
                    quats = gs_dict_j['quat'].float(),
                    scales = gs_dict_j['scale'].float(),
                    colors = gs_dict_j['color'].float(), # 0 - 1
                    opacities = gs_dict_j['opacity'].float(), # after sigmoid
                    viewmats = viewmat_i.float(),
                    Ks = K_i.float(),
                    width = width ,
                    height = height ,
                )
            colors_list.append(colors)
            
    if save_visualize:
        # print("colors shape:", colors.shape, colors.min(), colors.max(), colors.mean())
        # print("alphas shape:", alphas.shape, alphas.min(), alphas.max(), alphas.mean())
        colors_np = (colors_list[0][0].cpu().numpy() * 255).astype(np.uint8)
        from PIL import Image
        img = Image.fromarray(colors_np)
        corresponding_image_path = image_paths_batch[0][0]
        corresponding_image_base_name = corresponding_image_path.split("/")[-3] + '_'  + corresponding_image_path.split("/")[-1] 
        # img.save(f"./debug_/b{j}_view{view_i}_{corresponding_image_base_name}_render.png")
        img.save(target_render_save_name)
        # shutil.copy(corresponding_image_path, f"./debug_dino_vis/b{j}_view{view_i}_{corresponding_image_base_name}_input.png")
        # # img.save(f"./b{j}_render.png")
            
            

    return colors_list
      
