import torch.nn as nn
import torch_scatter

from pointcept.models.losses import build_criteria
from pointcept.models.losses.ibot_patch_loss import iBOTPatchLoss
from pointcept.models.losses.dino_clstoken_loss import DINOLoss
from pointcept.models.losses.sim_dino_clstoken_loss import MCRLoss
from pointcept.models.losses.sim_ibot_patch_loss import CosinePatchLoss

from pointcept.models.utils.structure import Point, Point
from .builder import MODELS, build_model

import torch
import numpy as np
from plyfile import PlyData, PlyElement
import os
from pathlib import Path
from pointcept.datasets.transform import GSGaussianBlurVoxelGPU
from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm
import time
from pointcept.models.utils import offset2batch
from torch_geometric.nn.pool import voxel_grid


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
        normalize=True,
        remove_last_layer=False,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(
            nlayers,
            in_dim,
            bottleneck_dim,
            hidden_dim=hidden_dim,
            use_bn=use_bn,
            bias=mlp_bias,
        )
        self.apply(self._init_weights)
        if not remove_last_layer:
            self.last_layer = weight_norm(
                nn.Linear(bottleneck_dim, out_dim, bias=False)
            )
            self.last_layer.weight_g.data.fill_(1)
        self.normalize = normalize
        self.remove_last_layer = remove_last_layer

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        # eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        if self.normalize:
            x = nn.functional.normalize(x, dim=-1, p=2, eps=1e-4)
        if not self.remove_last_layer:
            x = self.last_layer(x)
        return x


def _build_mlp(
    nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True
):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim, eps=1e-3))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim, eps=1e-3))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)


@MODELS.register_module()
class DefaultContrastiverSimDinoV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        local_crop_num=3,
        # do_mask=True, # do mask is prerequisite of do_ibot and enable_mae_loss
        do_ema=True,  # do ema is prerequisite of do_dino and do_ibot
        do_ibot=True,
        # do_dino=True,
        # do_ibot=True,
        # enable_local_ibot=False, # to save memory we always disable local_ibot like dinov2
        # enable_local_mae=False, # to save memory we always disable local_mae like dinov2
        share_mask=False,  # share mask then in sim ibot loss
        enable_mae_loss=False,
        mask_ratio_min_max=(0.1, 0.5),
        mask_sample_probability=0.5,
        dino_weight=1.0,
        ibot_weight=1.0,
        code_weight=1.0,
        mae_weight=1.0,
        # simple_dino=True, # using sim_dino instead of dinov2 original implementation, ibot always use old dinov2 implementation
        # head_n_prototypes= 8192,
        # head_n_prototypes_ibot = 256,
        mask_grid_size=0.2,  # 0.1 for indoor scene
        mask_type="patch",  # 'pixel' or 'patch'
    ):
        super().__init__()

        # sanity check
        # self.do_mask = do_mask
        self.do_ema = True
        # self.do_dino = True

        self.enable_mae_loss = enable_mae_loss
        # self.enable_local_mae = enable_local_mae
        self.dino_weight = dino_weight
        self.ibot_weight = ibot_weight
        self.code_weight = code_weight
        self.mae_weight = mae_weight
        self.num_layers = len(backbone["enc_depths"])
        self.local_crop_num = local_crop_num
        self.do_ibot = do_ibot
        # self.enable_local_ibot = enable_local_ibot
        self.mask_ratio_min_max = mask_ratio_min_max
        self.mask_sample_probability = mask_sample_probability
        self.mask_grid_size = mask_grid_size
        self.mask_type = mask_type

        # self.simple_dino = simple_dino
        # self.share_mask = share_mask # share mask is needed for simple dino ibot, which is not used now

        if do_ema:
            print("using ema, with teacher model and student model")
            # need to have do_dino or do_ibot
            # assert do_dino or do_ibot

        # start initialize
        backbone.do_mask = True
        if do_ema:
            self.backbone_teacher = build_model(backbone)
            # set teacher model requires_grad to False
            for param in self.backbone_teacher.parameters():
                param.requires_grad = False
            # otherwise we only have student model

        self.backbone_student = build_model(backbone)

        # considering
        # color range 0-1
        # normal range -1 - 1
        # quat range -1 - 1
        # scale range > 0
        # opacity range 0 - 1
        # coord range > 0
        # do we need to reconstruct the coord?
        # indoor, coord range too much, we need normalized loss or other loss like gs-mae
        # try without coord first
        if self.enable_mae_loss:
            self.mae_weight = mae_weight
            self.using_coord = True if backbone["in_channels"] == 17 else False
            self.mae_head = nn.Sequential(
                nn.Linear(backbone["dec_channels"][0], 32),
                nn.LayerNorm(32),
                nn.ReLU(inplace=True),
                nn.Linear(
                    32, 14
                ),  # backbone['in_channels'] = 17 (with coord), 14 (without coord) as input
            )
            # initialize
            for m in self.mae_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )

        # expansier
        self.dino_head = DINOHead(
            in_dim=backbone["enc_channels"][-1],
            out_dim=256,
            hidden_dim=2048,
            bottleneck_dim=256,
            nlayers=3,
            normalize=True,
            remove_last_layer=True,
        )

        self.dino_loss = MCRLoss(
            out_dim=256, expa_type=1, reduce_cov=0, eps=0.05, eps_end=-1, coeff=0.1
        )

        # we use seperate head here
        self.ibot_head = DINOHead(
            in_dim=backbone["dec_channels"][0],
            out_dim=32,
            hidden_dim=256,
            bottleneck_dim=32,
            nlayers=3,
            normalize=True,
            remove_last_layer=True,
        )

        self.ibot_patch_loss = CosinePatchLoss(patch_out_dim=32)

        # self.criteria = build_criteria(criteria)

    def save_views_to_ply(self, coords, color, scale, quat, opacity, filename):
        data_dict = {}
        data_dict["coord"] = coords.cpu().numpy()
        data_dict["color"] = ((color.cpu().numpy()) + 1) * 127.5
        data_dict["scale"] = scale.cpu().numpy()
        data_dict["quat"] = quat.cpu().numpy()
        data_dict["opacity"] = opacity.cpu().numpy()
        save_ply(data_dict, filename)

    def update_teacher(self, m=0.999):
        student_param_list = []
        teacher_param_list = []
        # to confirm the teacher is updated
        with torch.no_grad():
            for name, param in self.backbone_student.named_parameters():
                if not param.requires_grad:
                    continue
                if "mask_token" in name:
                    continue
                teacher_param = self.backbone_teacher.state_dict()[name]
                # if "embed" in name:
                #     old_embed_params = teacher_param.clone().cpu().detach().numpy()
                student_param_list.append(param)
                teacher_param_list.append(teacher_param)
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    def mask_generator(self, offset, view_origin_coord=None):  # also inclue patch
        with torch.no_grad():
            B = len(offset)  # totol samples
            offset_pad = nn.functional.pad(offset, (1, 0))
            n_samples_masked = int(np.ceil(B * self.mask_sample_probability))
            # probs = torch.linspace(*self.mask_ratio_min_max, n_samples_masked + 1)  # n_samples_masked have 0.1 - 0.5 probs
            upperbound = 0
            masks_list = []
            masks_weight_list = []
            # random sample indiceds 0 - to B with n_samples_masked
            mask_ge = torch.zeros(B, dtype=torch.long)
            mask_ge[:n_samples_masked] = 1
            mask_ge = mask_ge[torch.randperm(B)]

            if self.mask_type == "pixel":
                for i in range(0, B):
                    if mask_ge[i]:
                        token_num = offset_pad[i + 1] - offset_pad[i]
                        mask_rate = np.random.uniform(
                            self.mask_ratio_min_max[0], self.mask_ratio_min_max[1]
                        )
                        masked_sample_num = token_num * mask_rate
                        # prighynt("masked_sample_num", masked_sample_num, "token_num", token_num)
                        mask = torch.zeros(token_num, dtype=torch.bool).to(
                            offset.device
                        )
                        mask_weight = torch.ones(token_num, dtype=torch.float16).to(
                            offset.device
                        )
                        mask[: int(masked_sample_num)] = 1
                        mask = mask[torch.randperm(token_num)]
                        upperbound += 1
                        masks_list.append(mask)
                        mask_weight = mask_weight * (1 / (masked_sample_num))
                        masks_weight_list.append(mask_weight)
                    else:
                        token_num = offset_pad[i + 1] - offset_pad[i]
                        mask = torch.zeros(token_num, dtype=torch.bool).to(
                            offset.device
                        )
                        mask_weight = torch.ones(token_num, dtype=torch.float16).to(
                            offset.device
                        )
                        masks_list.append(mask)
                        masks_weight_list.append(mask_weight)

            elif self.mask_type == "patch":
                # union origin coord
                for i in range(0, B):
                    if mask_ge[i]:
                        batch_i_coord = view_origin_coord[
                            offset_pad[i] : offset_pad[i + 1]
                        ]
                        mask_patch_coord = batch_i_coord / (self.mask_grid_size)
                        mask_patch_grid_coord = torch.floor(mask_patch_coord).int()
                        min_coord = mask_patch_grid_coord.min(dim=0)[0]
                        mask_patch_grid_coord -= min_coord

                        mask_patch_cluster = voxel_grid(
                            pos=mask_patch_grid_coord,
                            size=1,
                            start=0,
                        )
                        unique, cluster, counts = torch.unique(
                            mask_patch_cluster,
                            sorted=True,
                            return_inverse=True,
                            return_counts=True,
                        )
                        patch_num = unique.shape[0]
                        patch_max_point = counts.max().item()
                        patch2point_map = cluster.new_zeros(patch_num, patch_max_point)
                        patch2point_mask = torch.lt(
                            torch.arange(patch_max_point).cuda().unsqueeze(0),
                            counts.unsqueeze(-1),
                        )
                        sorted_cluster_value, sorted_cluster_indices = torch.sort(
                            cluster
                        )
                        patch2point_map[patch2point_mask] = sorted_cluster_indices

                        patch_mask = torch.zeros(patch_num, dtype=torch.bool)
                        rand_perm = torch.randperm(patch_num)
                        mask_rate = np.random.uniform(
                            self.mask_ratio_min_max[0], self.mask_ratio_min_max[1]
                        )
                        mask_patch_num = int(patch_num * mask_rate)
                        # print("mask rate", mask_rate)
                        # print("mask_patch_num", mask_patch_num, "patch_num", patch_num)
                        patch_mask[rand_perm[0:mask_patch_num]] = True
                        point_mask = torch.zeros(mask_patch_coord.shape[0]).bool()
                        point_mask[
                            patch2point_map[patch_mask][patch2point_mask[patch_mask]]
                        ] = True

                        # print("point_mask", point_mask.sum(), point_mask.shape)

                        masks_list.append(point_mask.to(offset.device))
                        mask_weight = torch.ones(
                            point_mask.shape[0], dtype=torch.float16
                        ).to(offset.device)
                        mask_weight = mask_weight * (1 / mask_patch_num)
                        masks_weight_list.append(mask_weight)
                    else:
                        token_num = offset_pad[i + 1] - offset_pad[i]
                        mask = torch.zeros(token_num, dtype=torch.bool).to(
                            offset.device
                        )
                        mask_weight = torch.ones(token_num, dtype=torch.float16).to(
                            offset.device
                        )
                        masks_list.append(mask)
                        masks_weight_list.append(mask_weight)

            # for mask_weight_i in masks_weight_list:
            #     print("mask_weight_i", mask_weight_i.sum(), mask_weight_i.shape)
            masks_list = torch.cat(masks_list, dim=0)
            masks_weight_list = torch.cat(masks_weight_list, dim=0)
            # print("masks_list", masks_list.sum(), masks_list.shape)
            # print("masks_weight_list", masks_weight_list.sum(), masks_weight_list.shape)
            masks_weight_list = masks_weight_list[masks_list]

        return masks_list, masks_weight_list

    def calculate_sim_dino_loss_pool(
        self,
        teacher_dino_feat_global_pool,
        student_dino_feat_global_pool,
        student_dino_feat_local_pool_list,
    ):
        # teacher_dino_feat_global_pool: Batch x Crop x Feat_dim
        # student_dino_feat_global_pool: Batch x Crop x Feat_dim
        # student_dino_feat_local_pool_list [Batch x Crop x Feat_dim]
        # L2 normalize teacher_dino_feat_global_pool:

        # print("teacher_dino_feat_global_pool", teacher_dino_feat_global_pool.shape)
        # print("student_dino_feat_global_pool", student_dino_feat_global_pool.shape)
        teacher_dino_feat_global_pool = self.dino_head(
            teacher_dino_feat_global_pool
        )  # normalize in side
        teacher_dino_feat_global_pool = teacher_dino_feat_global_pool.permute(
            1, 0, 2
        )  # make the global crop number at begining
        teacher_dino_feat_global_pool = teacher_dino_feat_global_pool.reshape(
            -1, teacher_dino_feat_global_pool.shape[-1]
        )  # flatten the global crop number

        student_dino_feat_global_pool = self.dino_head(
            student_dino_feat_global_pool
        )  # normalize in side
        student_dino_feat_global_pool = student_dino_feat_global_pool.permute(
            1, 0, 2
        )  # make the global crop number at begining
        student_dino_feat_global_pool = student_dino_feat_global_pool.reshape(
            -1, student_dino_feat_global_pool.shape[-1]
        )  # flatten the global crop number

        z_student_local = torch.stack(
            student_dino_feat_local_pool_list, dim=1
        )  # we do not need z_teacher_local here
        z_student_local = self.dino_head(z_student_local)
        z_student_local = z_student_local.permute(
            1, 0, 2
        )  # make the global crop number at begining
        z_student_local = z_student_local.reshape(
            -1, z_student_local.shape[-1]
        )  # flatten the global crop number

        # print("student_dino_feat_global_pool", student_dino_feat_global_pool.min().item(), student_dino_feat_global_pool.max().item())
        # print("z_student_local", z_student_local.min().item(), z_student_local.max().item())
        # print("teacher_dino_feat_global_pool", teacher_dino_feat_global_pool.min().item(), teacher_dino_feat_global_pool.max().item())

        dino_crops_loss, dino_loss_dict = self.dino_loss(
            student_dino_feat_global_pool.chunk(2)
            + z_student_local.chunk(self.local_crop_num),
            teacher_dino_feat_global_pool.detach().chunk(2),
            no_diag=True,
        )
        # dino_crops_loss = torch.nn.functional.mse_loss(student_dino_feat_global_pool, teacher_dino_feat_global_pool.detach(), reduction='mean')
        # dino_crops_loss += torch.nn.functional.mse_loss(z_student_local[0], teacher_dino_feat_global_pool[0].detach(), reduction='mean')

        dino_loss_dict = {"dino_mcr_" + k: v for k, v in dino_loss_dict.items()}

        loss_dict = {}
        loss_dict.update(dino_loss_dict)
        loss_dict["sim_dino_crops_loss"] = dino_crops_loss
        return loss_dict

    def calculate_sim_dino_loss_patch(
        self,
        teacher_patch_feat_global_0_mask,
        teacher_patch_feat_global_1_mask,
        student_patch_feat_global_0_mask,
        student_patch_feat_global_1_mask,
        teacher_temp,
        global_mask_weight_0=None,
        global_mask_weight_1=None,
        view_num=1,
    ):
        # teacher_patch_feat_global_0_mask: Masked_token_num  x Feat_dim
        # teacher_patch_feat_global_1_mask: Masked_token_num' x Feat_dim
        # student_patch_feat_global_0_mask: Masked_token_num  x Feat_dim
        # student_patch_feat_global_1_mask: Masked_token_num' x Feat_dim
        # global_mask_weight_0: Masked_token_num
        # global_mask_weight_1: Masked_token_num'

        # local if needed

        teacher_patch_feat_global_mask = torch.cat(
            [teacher_patch_feat_global_0_mask, teacher_patch_feat_global_1_mask], dim=0
        )  # Masked_token_num x Feat_dim
        student_patch_feat_global_mask = torch.cat(
            [student_patch_feat_global_0_mask, student_patch_feat_global_1_mask], dim=0
        )  # Masked_token_num x Feat_dim
        global_mask = torch.cat(
            [global_mask_weight_0, global_mask_weight_1], dim=0
        )  # Masked_token_num

        teacher_patch_feat_global_mask = self.ibot_head(
            teacher_patch_feat_global_mask
        )  # Masked_token_num x prototype_dim
        student_patch_feat_global_mask = self.ibot_head(
            student_patch_feat_global_mask
        )  # Masked_token_num x prototype_dim

        ibot_patch_loss = self.ibot_patch_loss.forward_masked(
            student_patch_feat_global_mask,
            teacher_patch_feat_global_mask.detach(),
            masks_weight=global_mask,
            view_nums=view_num,
        )
        ibot_patch_loss, ibot_loss_dict = ibot_patch_loss
        ibot_loss_dict = {"ibot_" + k: v for k, v in ibot_loss_dict.items()}
        loss_dict = {}
        loss_dict.update(ibot_loss_dict)
        loss_dict["sim_ibot_patch_loss"] = ibot_patch_loss
        return loss_dict

    def forward(self, input_dict, teacher_temp, visualize=False):
        # input results
        # data_dict dict_keys(['coord', 'color', 'opacity', 'scale', 'quat', 'instance', 'segment', 'name', 'global_crop1_coord', 'global_crop1_color', 'global_crop1_scale', 'global_crop1_quat', 'global_crop1_opacity', 'global_crop1_grid_coord', 'global_crop2_coord', 'global_crop2_color', 'global_crop2_scale', 'global_crop2_quat', 'global_crop2_opacity', 'global_crop2_grid_coord', 'local_crop0_coord', 'local_crop0_color', 'local_crop0_scale', 'local_crop0_quat', 'local_crop0_opacity', 'local_crop0_grid_coord', 'local_crop1_coord', 'local_crop1_color', 'local_crop1_scale', 'local_crop1_quat', 'local_crop1_opacity', 'local_crop1_grid_coord', 'local_crop2_coord', 'local_crop2_color', 'local_crop2_scale', 'local_crop2_quat', 'local_crop2_opacity', 'local_crop2_grid_coord', 'local_crop3_coord', 'local_crop3_color', 'local_crop3_scale', 'local_crop3_quat', 'local_crop3_opacity', 'local_crop3_grid_coord'])
        # view1_coord = input_dict["view1_coord"]
        # view1_feat = input_dict["view1_feat"]
        # view1_offset = input_dict["view1_offset"]
        # print("input offset global0", input_dict["global_crop0_offset"])
        # print("input offset global1", input_dict["global_crop1_offset"])
        # print("input offset local0", input_dict["local_crop0_offset"])
        # print("input offset local1", input_dict["local_crop1_offset"])

        print(
            "globel view pts num 0", input_dict["global_crop0_offset"][-1]
        )  # dtype torch.float32 # "dtype", input_dict["global_crop0_feat"].dtype
        # float32 again, so after transform, it become fp32
        # print("batci size per gpu", input_dict["global_crop0_offset"].shape[0])
        # globel view pts num tensor(3276800, device='cuda:0')
        if self.local_crop_num > 0:
            print("local view pts num", input_dict["local_crop0_offset"][-1])

        # input debug
        if visualize:
            time_save_str = time.strftime(
                "%Y-%m-%d-%H-%M-%S", time.localtime(time.time())
            )
            data_save_path = f"exp/augs_test_gs/unmasked/{time_save_str}/"
            Path(data_save_path).mkdir(parents=True, exist_ok=True)
            # debug strange scale size here
            print(
                "global_crop0_scale",
                input_dict["global_crop0_feat"][:, 3:6].min(),
                input_dict["global_crop0_feat"][:, 3:6].max(),
            )
            print(
                "global_crop1_scale",
                input_dict["global_crop1_feat"][:, 3:6].min(),
                input_dict["global_crop1_feat"][:, 3:6].max(),
            )
            # oonly take the first offset
            global_crop0_offset_0 = input_dict["global_crop0_offset"][0]
            global_crop_0_data = dict(
                coord=input_dict["global_crop0_coord"][:global_crop0_offset_0],
                color=(input_dict["global_crop0_feat"][:global_crop0_offset_0, :3] + 1)
                * 127.5,
                scale=input_dict["global_crop0_feat"][:global_crop0_offset_0, 3:6],
                quat=input_dict["global_crop0_feat"][:global_crop0_offset_0, 6:10],
                opacity=input_dict["global_crop0_feat"][:global_crop0_offset_0, 10],
                normal=input_dict["global_crop0_feat"][:global_crop0_offset_0, 11:14],
                offset=input_dict["global_crop0_offset"],
                grid_coord=input_dict["global_crop0_grid_coord"],
            )

            for key_i in global_crop_0_data.keys():
                # go to cpu and numpy
                global_crop_0_data[key_i] = global_crop_0_data[key_i].cpu().numpy()

            save_ply(global_crop_0_data, f"{data_save_path}/global_crop_0.ply")
            global_crop1_offset_0 = input_dict["global_crop1_offset"][0]
            global_crop_1_data = dict(
                coord=input_dict["global_crop1_coord"][:global_crop1_offset_0],
                color=(input_dict["global_crop1_feat"][:global_crop1_offset_0, :3] + 1)
                * 127.5,
                scale=input_dict["global_crop1_feat"][:global_crop1_offset_0, 3:6],
                quat=input_dict["global_crop1_feat"][:global_crop1_offset_0, 6:10],
                opacity=input_dict["global_crop1_feat"][:global_crop1_offset_0, 10],
                normal=input_dict["global_crop1_feat"][:global_crop1_offset_0, 11:14],
                offset=input_dict["global_crop1_offset"],
                grid_coord=input_dict["global_crop1_grid_coord"],
            )
            for key_i in global_crop_1_data.keys():
                # go to cpu and numpy
                global_crop_1_data[key_i] = global_crop_1_data[key_i].cpu().numpy()

            save_ply(global_crop_1_data, f"{data_save_path}/global_crop_1.ply")
            for i in range(self.local_crop_num):
                local_crop_data_offset_0 = input_dict[f"local_crop{i}_offset"][0]
                local_crop_data = dict(
                    coord=input_dict[f"local_crop{i}_coord"][:local_crop_data_offset_0],
                    color=(
                        input_dict[f"local_crop{i}_feat"][:local_crop_data_offset_0, :3]
                        + 1
                    )
                    * 127.5,
                    scale=input_dict[f"local_crop{i}_feat"][
                        :local_crop_data_offset_0, 3:6
                    ],
                    quat=input_dict[f"local_crop{i}_feat"][
                        :local_crop_data_offset_0, 6:10
                    ],
                    opacity=input_dict[f"local_crop{i}_feat"][
                        :local_crop_data_offset_0, 10
                    ],
                    normal=input_dict[f"local_crop{i}_feat"][
                        :local_crop_data_offset_0, 11:14
                    ],
                    offset=input_dict[f"local_crop{i}_offset"],
                    grid_coord=input_dict[f"local_crop{i}_grid_coord"],
                )
                for key_i in local_crop_data.keys():
                    # go to cpu and numpy
                    local_crop_data[key_i] = local_crop_data[key_i].cpu().numpy()
                save_ply(local_crop_data, f"{data_save_path}/local_crop_{i}.ply")

        # prepare input
        global_crop0_data_dict = dict(
            coord=input_dict["global_crop0_coord"],
            feat=input_dict["global_crop0_feat"],
            offset=input_dict["global_crop0_offset"],
            grid_coord=input_dict["global_crop0_grid_coord"],
        )
        global_crop1_data_dict = dict(
            coord=input_dict["global_crop1_coord"],
            feat=input_dict["global_crop1_feat"],
            offset=input_dict["global_crop1_offset"],
            grid_coord=input_dict["global_crop1_grid_coord"],
        )
        local_crop_data_dict_list = []
        for i in range(self.local_crop_num):
            local_crop_data_dict = dict(
                coord=input_dict[f"local_crop{i}_coord"].half(),
                feat=input_dict[f"local_crop{i}_feat"].half(),
                offset=input_dict[f"local_crop{i}_offset"],
                grid_coord=input_dict[f"local_crop{i}_grid_coord"],
            )
            local_crop_data_dict_list.append(local_crop_data_dict)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # get teacher output funciton
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        def get_teacher_output(
            global_view_0,
            global_view_1,
            global_mask_0=None,
            global_mask_1=None,
            local_views=None,
            local_masks=None,
        ):
            # teacher do not need mask
            # 1: maximal is 1228800
            # 2: known when achieve
            # chunkify the backbone process 1827558
            batch_size = global_view_0["offset"].shape[0]

            if False:  # batch_size >=2: #global_view_0['offset'][-1] > (204800 * batch_size) * 0.5:
                #  print("enable chunkify")
                # chunk to two part and process
                # print some basic information
                # print('global_view_0', global_view_0.keys())
                # print("global_view_0 offset", global_view_0['offset'])
                # global_view_0 dict_keys(['coord', 'feat', 'offset', 'grid_coord'])
                # global_view_0 offset tensor([ 204800,  409600,  587354,  792154,  880392, 1085192], device='cuda:0')
                # global_view_0_chunk_0 tensor([204800, 409600, 587354, 792154], device='cuda:0')

                global_view_0_chunk_0 = {}
                global_view_0_chunk_1 = {}
                chunk0_batch_size = int(batch_size / 2)
                chunk1_batch_size = batch_size - chunk0_batch_size
                chunk0_offset = global_view_0["offset"][:chunk0_batch_size]
                chunk1_offset = (
                    global_view_0["offset"][chunk0_batch_size:] - chunk0_offset[-1]
                )
                global_view_0_chunk_0["offset"] = chunk0_offset
                global_view_0_chunk_1["offset"] = chunk1_offset
                global_view_0_chunk_0["coord"] = global_view_0["coord"][
                    : chunk0_offset[-1]
                ]
                global_view_0_chunk_1["coord"] = global_view_0["coord"][
                    chunk0_offset[-1] :
                ]
                global_view_0_chunk_0["feat"] = global_view_0["feat"][
                    : chunk0_offset[-1]
                ]
                global_view_0_chunk_1["feat"] = global_view_0["feat"][
                    chunk0_offset[-1] :
                ]
                global_view_0_chunk_0["grid_coord"] = global_view_0["grid_coord"][
                    : chunk0_offset[-1]
                ]
                global_view_0_chunk_1["grid_coord"] = global_view_0["grid_coord"][
                    chunk0_offset[-1] :
                ]
                global_mask_0_chunk_0 = global_mask_0[: chunk0_offset[-1]]
                global_mask_0_chunk_1 = global_mask_0[
                    chunk0_offset[-1] :
                ]  # for mask we do not need to minus the offset

                # print("global_view_0_chunk_0", global_view_0_chunk_0['offset'])
                # print("global_view_0_chunk_1", global_view_0_chunk_1['offset'])

                global_view_1_chunk_0 = {}
                global_view_1_chunk_1 = {}
                chunk0_offset = global_view_1["offset"][:chunk0_batch_size]
                chunk1_offset = (
                    global_view_1["offset"][chunk0_batch_size:] - chunk0_offset[-1]
                )
                global_view_1_chunk_0["offset"] = chunk0_offset
                global_view_1_chunk_1["offset"] = chunk1_offset
                global_view_1_chunk_0["coord"] = global_view_1["coord"][
                    : chunk0_offset[-1]
                ]
                global_view_1_chunk_1["coord"] = global_view_1["coord"][
                    chunk0_offset[-1] :
                ]
                global_view_1_chunk_0["feat"] = global_view_1["feat"][
                    : chunk0_offset[-1]
                ]
                global_view_1_chunk_1["feat"] = global_view_1["feat"][
                    chunk0_offset[-1] :
                ]
                global_view_1_chunk_0["grid_coord"] = global_view_1["grid_coord"][
                    : chunk0_offset[-1]
                ]
                global_view_1_chunk_1["grid_coord"] = global_view_1["grid_coord"][
                    chunk0_offset[-1] :
                ]
                global_mask_1_chunk_0 = (
                    global_mask_1[: chunk0_offset[-1]]
                    if global_mask_1 is not None
                    else None
                )
                global_mask_1_chunk_1 = global_mask_1[
                    chunk0_offset[-1] :
                ]  # for mask we do not need to minus the offset

                # for key_i in global_view_0_chunk_0.keys():
                #     print("global_view_0_chunk_0", key_i, global_view_0_chunk_0[key_i].shape)
                # print("global_view_0_chunk_0 offset", global_view_0_chunk_0['offset'])

                # for key_i in global_view_0_chunk_1.keys():
                #     print("global_view_0_chunk_1", key_i, global_view_0_chunk_1[key_i].shape)
                # print("global_view_0_chunk_1 offset", global_view_0_chunk_1['offset'])

                # raise ValueError("chunking is not supported now, please disable it")
                (
                    teacher_dino_feat_global_0_chunk_0,
                    teacher_patch_feat_global_0_chunk_0,
                ) = self.backbone_teacher(Point(global_view_0_chunk_0), return_dec=True)
                (
                    teacher_dino_feat_global_0_chunk_1,
                    teacher_patch_feat_global_0_chunk_1,
                ) = self.backbone_teacher(Point(global_view_0_chunk_1), return_dec=True)
                (
                    teacher_dino_feat_global_1_chunk_0,
                    teacher_patch_feat_global_1_chunk_0,
                ) = self.backbone_teacher(Point(global_view_1_chunk_0), return_dec=True)
                (
                    teacher_dino_feat_global_1_chunk_1,
                    teacher_patch_feat_global_1_chunk_1,
                ) = self.backbone_teacher(Point(global_view_1_chunk_1), return_dec=True)
                # teacher_dino_feat_global_0 = torch.cat([teacher_dino_feat_global_0_chunk_0.feat, teacher_dino_feat_global_0_chunk_1.feat], dim=0)
                # teacher_patch_feat_global_0 = torch.cat([teacher_patch_feat_global_0_chunk_0.feat, teacher_patch_feat_global_0_chunk_1.feat], dim=0)
                # teacher_dino_feat_global_1 = torch.cat([teacher_dino_feat_global_1_chunk_0.feat, teacher_dino_feat_global_1_chunk_1.feat], dim=0)
                # teacher_patch_feat_global_1 = torch.cat([teacher_patch_feat_global_1_chunk_0.feat, teacher_patch_feat_global_1_chunk_1.feat], dim=0)
                # when pooling  the feature with mean, we can mean again the chunk
                indptr = nn.functional.pad(
                    teacher_dino_feat_global_0_chunk_0.offset, (1, 0)
                )

                teacher_dino_feat_global_0_chunk_0_pool = torch_scatter.segment_csr(
                    src=teacher_dino_feat_global_0_chunk_0.feat,
                    indptr=nn.functional.pad(
                        teacher_dino_feat_global_0_chunk_0.offset, (1, 0)
                    ),
                    reduce="mean",
                )  # from chunk0 in global view 0
                # print("teacher_dino_feat_global_0_chunk_0_pool", teacher_dino_feat_global_0_chunk_0_pool.shape)

                teacher_dino_feat_global_0_chunk_1_pool = torch_scatter.segment_csr(
                    src=teacher_dino_feat_global_0_chunk_1.feat,
                    indptr=nn.functional.pad(
                        teacher_dino_feat_global_0_chunk_1.offset, (1, 0)
                    ),
                    reduce="mean",
                )  # from chunk1 in global view 0
                # print("teacher_dino_feat_global_0_chunk_1_pool", teacher_dino_feat_global_0_chunk_1_pool.shape)
                teacher_dino_feat_global_1_chunk_0_pool = torch_scatter.segment_csr(
                    src=teacher_dino_feat_global_1_chunk_0.feat,
                    indptr=nn.functional.pad(
                        teacher_dino_feat_global_1_chunk_0.offset, (1, 0)
                    ),
                    reduce="mean",
                )
                teacher_dino_feat_global_1_chunk_1_pool = torch_scatter.segment_csr(
                    src=teacher_dino_feat_global_1_chunk_1.feat,
                    indptr=nn.functional.pad(
                        teacher_dino_feat_global_1_chunk_1.offset, (1, 0)
                    ),
                    reduce="mean",
                )
                teacher_dino_feat_global_0_pool = torch.cat(
                    [
                        teacher_dino_feat_global_0_chunk_0_pool,
                        teacher_dino_feat_global_0_chunk_1_pool,
                    ],
                    dim=0,
                )  # Batch x Crop x Feat_dim
                teacher_dino_feat_global_1_pool = torch.cat(
                    [
                        teacher_dino_feat_global_1_chunk_0_pool,
                        teacher_dino_feat_global_1_chunk_1_pool,
                    ],
                    dim=0,
                )  # Batch x Crop x Feat_dim
                # now we proceess the patch feature
                # for patch feature we need offset
                mask_teacher_patch_feat_global_0_chunk_0 = (
                    teacher_patch_feat_global_0_chunk_0.feat[global_mask_0_chunk_0]
                )  # M0xC
                mask_teacher_patch_feat_global_0_chunk_1 = (
                    teacher_patch_feat_global_0_chunk_1.feat[global_mask_0_chunk_1]
                )  # M0xC
                mask_teacher_patch_feat_global_1_chunk_0 = (
                    teacher_patch_feat_global_1_chunk_0.feat[global_mask_1_chunk_0]
                )  # M1xC
                mask_teacher_patch_feat_global_1_chunk_1 = (
                    teacher_patch_feat_global_1_chunk_1.feat[global_mask_1_chunk_1]
                )  # M1xC
                teacher_patch_feat_global_0_mask = torch.cat(
                    [
                        mask_teacher_patch_feat_global_0_chunk_0,
                        mask_teacher_patch_feat_global_0_chunk_1,
                    ],
                    dim=0,
                )  # M0xC
                teacher_patch_feat_global_1_mask = torch.cat(
                    [
                        mask_teacher_patch_feat_global_1_chunk_0,
                        mask_teacher_patch_feat_global_1_chunk_1,
                    ],
                    dim=0,
                )  # M1xC

            else:
                # input is fp16
                # first we check no NaN in the input

                # print("global_view_0 coord", global_view_0['coord'].min().item(), global_view_0['coord'].max().item())
                # print("global_view_0 feat", global_view_0['feat'].min().item(), global_view_0['feat'].max().item())
                # print("global_view_0 grid_coord", global_view_0['grid_coord'].min().item(), global_view_0['grid_coord'].max().item())
                # print("global_view_0 offset", global_view_0['offset'])

                teacher_dino_feat_global_0, teacher_patch_feat_global_0 = (
                    self.backbone_teacher(Point(global_view_0), return_dec=True)
                )
                teacher_dino_feat_global_1, teacher_patch_feat_global_1 = (
                    self.backbone_teacher(Point(global_view_1), return_dec=True)
                )
                # print("teacher_dino_feat_global_0.offset", teacher_dino_feat_global_0.offset)
                # after backbone to fp32
                # check if teacher_dino_feat_global_0 have NaN
                # if torch.isnan(teacher_dino_feat_global_0.feat).any():
                #     print("global_view_0 grid_coord", global_view_0['grid_coord'].min().item(), global_view_0['grid_coord'].max().item())
                #     print("global_view_0 offset", global_view_0['offset'])
                #     raise ValueError("teacher_dino_feat_global_0 has NaN")
                # global_view_0 coord -2.598296880722046 4.167304039001465
                # global_view_0 feat -0.9999999403953552 13.490856170654297
                # global_view_0 grid_coord 0 260
                # global_view_0 offset tensor([116845, 238795, 443595], device='cuda:0')
                # global_view_0 grid_coord 0 260
                # global_view_0 offset tensor([116845, 238795, 443595], device='cuda:0')

                # print("teacher_patch_feat_global_0", teacher_patch_feat_global_0.feat.dtype) # torch.float32
                teacher_dino_feat_global_0_pool = torch_scatter.segment_csr(
                    src=teacher_dino_feat_global_0.feat,
                    indptr=nn.functional.pad(teacher_dino_feat_global_0.offset, (1, 0)),
                    reduce="mean",
                )

                teacher_dino_feat_global_1_pool = torch_scatter.segment_csr(
                    src=teacher_dino_feat_global_1.feat,
                    indptr=nn.functional.pad(teacher_dino_feat_global_1.offset, (1, 0)),
                    reduce="mean",
                )
                assert global_mask_0 is not None and global_mask_1 is not None
                teacher_patch_feat_global_0_mask = teacher_patch_feat_global_0.feat[
                    global_mask_0
                ]  # M0xC
                teacher_patch_feat_global_1_mask = teacher_patch_feat_global_1.feat[
                    global_mask_1
                ]  # M1xC

            teacher_dino_feat_global_pool = torch.cat(
                [
                    teacher_dino_feat_global_0_pool.unsqueeze(1),
                    teacher_dino_feat_global_1_pool.unsqueeze(1),
                ],
                dim=1,
            )  # Batch x Crop x Feat_dim

            # can only concat if mask is the same

            return (
                teacher_dino_feat_global_pool,
                teacher_patch_feat_global_0_mask,
                teacher_patch_feat_global_1_mask,
            )

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # get student output funciton
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        def get_student_output(
            global_view_0,
            global_view_1,
            local_views,
            global_mask_0=None,
            global_mask_1=None,
            local_masks=None,
        ):
            batch_size = global_view_0["offset"].shape[0]

            if False:  # batch_size >= 2:#  global_view_0['offset'][-1] > (204800 * batch_size) * 0.5:
                # print("enable chunking")
                # chunk to two part and process
                # print some basic information
                # print('global_view_0', global_view_0.keys())
                # print("global_view_0 offset", global_view_0['offset'])
                # global_view_0 dict_keys(['coord', 'feat', 'offset', 'grid_coord'])
                # global_view_0 offset tensor([ 204800,  409600,  587354,  792154,  880392, 1085192], device='cuda:0')
                # global_view_0_chunk_0 tensor([204800, 409600, 587354, 792154], device='cuda:0')
                # global_view_0_chunk_1 tensor([     0,  88238, 293038], device='cuda:0')

                global_view_0_chunk_0 = {}
                global_view_0_chunk_1 = {}
                chunk0_batch_size = int(batch_size / 2)
                chunk1_batch_size = batch_size - chunk0_batch_size
                chunk0_offset = global_view_0["offset"][:chunk0_batch_size]
                chunk1_offset = (
                    global_view_0["offset"][chunk0_batch_size:] - chunk0_offset[-1]
                )
                global_view_0_chunk_0["offset"] = chunk0_offset
                global_view_0_chunk_1["offset"] = chunk1_offset
                global_view_0_chunk_0["coord"] = global_view_0["coord"][
                    : chunk0_offset[-1]
                ]
                global_view_0_chunk_1["coord"] = global_view_0["coord"][
                    chunk0_offset[-1] :
                ]
                global_view_0_chunk_0["feat"] = global_view_0["feat"][
                    : chunk0_offset[-1]
                ]
                global_view_0_chunk_1["feat"] = global_view_0["feat"][
                    chunk0_offset[-1] :
                ]
                global_view_0_chunk_0["grid_coord"] = global_view_0["grid_coord"][
                    : chunk0_offset[-1]
                ]
                global_view_0_chunk_1["grid_coord"] = global_view_0["grid_coord"][
                    chunk0_offset[-1] :
                ]
                global_mask_0_chunk_0 = global_mask_0[: chunk0_offset[-1]]
                global_mask_0_chunk_1 = global_mask_0[
                    chunk0_offset[-1] :
                ]  # for mask we do not need to minus the offset

                # print("global_view_0_chunk_0", global_view_0_chunk_0['offset'])
                # print("global_view_0_chunk_1", global_view_0_chunk_1['offset'])
                global_view_1_chunk_0 = {}
                global_view_1_chunk_1 = {}
                chunk0_offset = global_view_1["offset"][:chunk0_batch_size]
                chunk1_offset = (
                    global_view_1["offset"][chunk0_batch_size:] - chunk0_offset[-1]
                )
                global_view_1_chunk_0["offset"] = chunk0_offset
                global_view_1_chunk_1["offset"] = chunk1_offset
                global_view_1_chunk_0["coord"] = global_view_1["coord"][
                    : chunk0_offset[-1]
                ]
                global_view_1_chunk_1["coord"] = global_view_1["coord"][
                    chunk0_offset[-1] :
                ]
                global_view_1_chunk_0["feat"] = global_view_1["feat"][
                    : chunk0_offset[-1]
                ]
                global_view_1_chunk_1["feat"] = global_view_1["feat"][
                    chunk0_offset[-1] :
                ]
                global_view_1_chunk_0["grid_coord"] = global_view_1["grid_coord"][
                    : chunk0_offset[-1]
                ]
                global_view_1_chunk_1["grid_coord"] = global_view_1["grid_coord"][
                    chunk0_offset[-1] :
                ]
                global_mask_1_chunk_0 = global_mask_1[: chunk0_offset[-1]]
                global_mask_1_chunk_1 = global_mask_1[
                    chunk0_offset[-1] :
                ]  # for mask we do not need to minus the offset

                (
                    student_dino_feat_global_0_chunk_0,
                    student_patch_feat_global_0_chunk_0,
                ) = self.backbone_student(Point(global_view_0_chunk_0), return_dec=True)
                (
                    student_dino_feat_global_0_chunk_1,
                    student_patch_feat_global_0_chunk_1,
                ) = self.backbone_student(Point(global_view_0_chunk_1), return_dec=True)
                (
                    student_dino_feat_global_1_chunk_0,
                    student_patch_feat_global_1_chunk_0,
                ) = self.backbone_student(Point(global_view_1_chunk_0), return_dec=True)
                (
                    student_dino_feat_global_1_chunk_1,
                    student_patch_feat_global_1_chunk_1,
                ) = self.backbone_student(Point(global_view_1_chunk_1), return_dec=True)

                # when pooling  the feature with mean, we can mean again the chunk
                student_dino_feat_global_0_chunk_0_pool = torch_scatter.segment_csr(
                    src=student_dino_feat_global_0_chunk_0.feat,
                    indptr=nn.functional.pad(
                        student_dino_feat_global_0_chunk_0.offset, (1, 0)
                    ),
                    reduce="mean",
                )  # from chunk0 in global view 0
                # print("teacher_dino_feat_global_0_chunk_0_pool", teacher_dino_feat_global_0_chunk_0_pool.shape)
                student_dino_feat_global_0_chunk_1_pool = torch_scatter.segment_csr(
                    src=student_dino_feat_global_0_chunk_1.feat,
                    indptr=nn.functional.pad(
                        student_dino_feat_global_0_chunk_1.offset, (1, 0)
                    ),
                    reduce="mean",
                )  # from chunk1 in global view 0
                # print("teacher_dino_feat_global_0_chunk_1_pool", teacher_dino_feat_global_0_chunk_1_pool.shape)
                student_dino_feat_global_1_chunk_0_pool = torch_scatter.segment_csr(
                    src=student_dino_feat_global_1_chunk_0.feat,
                    indptr=nn.functional.pad(
                        student_dino_feat_global_1_chunk_0.offset, (1, 0)
                    ),
                    reduce="mean",
                )
                student_dino_feat_global_1_chunk_1_pool = torch_scatter.segment_csr(
                    src=student_dino_feat_global_1_chunk_1.feat,
                    indptr=nn.functional.pad(
                        student_dino_feat_global_1_chunk_1.offset, (1, 0)
                    ),
                    reduce="mean",
                )
                student_dino_feat_global_0_pool = torch.cat(
                    [
                        student_dino_feat_global_0_chunk_0_pool,
                        student_dino_feat_global_0_chunk_1_pool,
                    ],
                    dim=0,
                )  # Batch x Crop x Feat_dim
                student_dino_feat_global_1_pool = torch.cat(
                    [
                        student_dino_feat_global_1_chunk_0_pool,
                        student_dino_feat_global_1_chunk_1_pool,
                    ],
                    dim=0,
                )  # Batch x Crop x Feat_dim
                # teacher_dino_feat_global_0_pool = (teacher_dino_feat_global_0_chunk_0_pool + teacher_dino_feat_global_0_chunk_1_pool) / 2
                # teacher_dino_feat_global_1_pool = (teacher_dino_feat_global_1_chunk_0_pool + teacher_dino_feat_global_1_chunk_1_pool) / 2
                # patch

                # now we proceess the patch feature
                mask_student_patch_feat_global_0_chunk_0 = (
                    student_patch_feat_global_0_chunk_0.feat[global_mask_0_chunk_0]
                )
                mask_student_patch_feat_global_0_chunk_1 = (
                    student_patch_feat_global_0_chunk_1.feat[global_mask_0_chunk_1]
                )
                mask_student_patch_feat_global_1_chunk_0 = (
                    student_patch_feat_global_1_chunk_0.feat[global_mask_1_chunk_0]
                )
                mask_student_patch_feat_global_1_chunk_1 = (
                    student_patch_feat_global_1_chunk_1.feat[global_mask_1_chunk_1]
                )

                student_patch_feat_global_0_mask = torch.cat(
                    [
                        mask_student_patch_feat_global_0_chunk_0,
                        mask_student_patch_feat_global_0_chunk_1,
                    ],
                    dim=0,
                )  # M0xC
                student_patch_feat_global_1_mask = torch.cat(
                    [
                        mask_student_patch_feat_global_1_chunk_0,
                        mask_student_patch_feat_global_1_chunk_1,
                    ],
                    dim=0,
                )  # M1xC

            else:
                student_dino_feat_global_0, student_patch_feat_global_0 = (
                    self.backbone_student(
                        Point(global_view_0), mask=global_mask_0, return_dec=True
                    )
                )
                student_dino_feat_global_1, student_patch_feat_global_1 = (
                    self.backbone_student(
                        Point(global_view_1), mask=global_mask_1, return_dec=True
                    )
                )

                # print("student_dino_feat_global_0", student_dino_feat_global_0.offset)
                student_dino_feat_global_0_pool = torch_scatter.segment_csr(
                    src=student_dino_feat_global_0.feat,
                    indptr=nn.functional.pad(student_dino_feat_global_0.offset, (1, 0)),
                    reduce="mean",
                )

                # print("student_dino_feat_global_1", student_dino_feat_global_1.offset)
                student_dino_feat_global_1_pool = torch_scatter.segment_csr(
                    src=student_dino_feat_global_1.feat,
                    indptr=nn.functional.pad(student_dino_feat_global_1.offset, (1, 0)),
                    reduce="mean",
                )

                assert global_mask_0 is not None and global_mask_1 is not None
                student_patch_feat_global_0_mask = student_patch_feat_global_0.feat[
                    global_mask_0
                ]
                student_patch_feat_global_1_mask = student_patch_feat_global_1.feat[
                    global_mask_1
                ]

            student_dino_feat_global_pool = torch.cat(
                [
                    student_dino_feat_global_0_pool.unsqueeze(1),
                    student_dino_feat_global_1_pool.unsqueeze(1),
                ],
                dim=1,
            )

            # disable local ibot, here the local input is not masked, and return dec is False
            student_dino_feat_local_pool_list = []
            for i in range(self.local_crop_num):
                student_dino_feat_local_i, _ = self.backbone_student(
                    Point(local_views[i]), return_dec=False
                )
                student_dino_feat_local_i_pool = torch_scatter.segment_csr(
                    src=student_dino_feat_local_i.feat,
                    indptr=nn.functional.pad(student_dino_feat_local_i.offset, (1, 0)),
                    reduce="mean",
                )

                student_dino_feat_local_pool_list.append(student_dino_feat_local_i_pool)

            return (
                student_dino_feat_global_pool,
                student_patch_feat_global_0_mask,
                student_patch_feat_global_1_mask,
                student_dino_feat_local_pool_list,
            )

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # main function contintuse
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # view from global

        global_mask_0, global_mask_0_weight = self.mask_generator(
            global_crop0_data_dict["offset"], global_crop0_data_dict["grid_coord"]
        )
        global_mask_1, global_mask_1_weight = self.mask_generator(
            global_crop1_data_dict["offset"], global_crop1_data_dict["grid_coord"]
        )
        # by default we use different mask for two global view

        # do not lmask student locally

        # mask debug
        if visualize:
            print(
                "mask num global0", global_mask_0.sum(), "out of", global_mask_0.shape
            )
            print(
                "mask num global1", global_mask_1.sum(), "out of", global_mask_1.shape
            )
            global_mask_0_reverse = ~global_mask_0
            global_mask_1_reverse = ~global_mask_1
            data_save_path = f"exp/augs_test_gs/masked/{time_save_str}/"
            Path(data_save_path).mkdir(parents=True, exist_ok=True)
            global_crop_0_offset_0 = global_crop0_data_dict["offset"][0]
            global_crop_0_data = dict(
                coord=input_dict["global_crop0_coord"][:global_crop_0_offset_0][
                    global_mask_0_reverse[:global_crop_0_offset_0]
                ],
                color=(
                    input_dict["global_crop0_feat"][:global_crop_0_offset_0][
                        global_mask_0_reverse[:global_crop_0_offset_0], :3
                    ]
                    + 1
                )
                * 127.5,
                scale=input_dict["global_crop0_feat"][:global_crop_0_offset_0][
                    global_mask_0_reverse[:global_crop_0_offset_0], 3:6
                ],
                quat=input_dict["global_crop0_feat"][:global_crop_0_offset_0][
                    global_mask_0_reverse[:global_crop_0_offset_0], 6:10
                ],
                opacity=input_dict["global_crop0_feat"][:global_crop_0_offset_0][
                    global_mask_0_reverse[:global_crop_0_offset_0], 10
                ],
                normal=input_dict["global_crop0_feat"][:global_crop_0_offset_0][
                    global_mask_0_reverse[:global_crop_0_offset_0], 11:14
                ],
            )
            for key_i in global_crop_0_data.keys():
                # go to cpu and numpy
                global_crop_0_data[key_i] = global_crop_0_data[key_i].cpu().numpy()
            save_ply(global_crop_0_data, f"{data_save_path}/global_crop_0_mask.ply")
            global_crop_1_offset_0 = global_crop1_data_dict["offset"][0]
            global_crop_1_data = dict(
                coord=input_dict["global_crop1_coord"][:global_crop_1_offset_0][
                    global_mask_1_reverse[:global_crop_1_offset_0]
                ],
                color=(
                    input_dict["global_crop1_feat"][:global_crop_1_offset_0][
                        global_mask_1_reverse[:global_crop_1_offset_0], :3
                    ]
                    + 1
                )
                * 127.5,
                scale=input_dict["global_crop1_feat"][:global_crop_1_offset_0][
                    global_mask_1_reverse[:global_crop_1_offset_0], 3:6
                ],
                quat=input_dict["global_crop1_feat"][:global_crop_1_offset_0][
                    global_mask_1_reverse[:global_crop_1_offset_0], 6:10
                ],
                opacity=input_dict["global_crop1_feat"][:global_crop_1_offset_0][
                    global_mask_1_reverse[:global_crop_1_offset_0], 10
                ],
                normal=input_dict["global_crop1_feat"][:global_crop_1_offset_0][
                    global_mask_1_reverse[:global_crop_1_offset_0], 11:14
                ],
            )
            # global_crop_1_data = dict(
            #     coord=input_dict["global_crop1_coord"][global_mask_1_reverse],
            #     color = (input_dict["global_crop1_feat"][global_mask_1_reverse, :3]+1)*127.5,
            #     scale = input_dict["global_crop1_feat"][global_mask_1_reverse, 3:6],
            #     quat = input_dict["global_crop1_feat"][global_mask_1_reverse, 6:10],
            #     opacity = input_dict["global_crop1_feat"][global_mask_1_reverse, 10],
            #     normal = input_dict["global_crop1_feat"][global_mask_1_reverse, 11:14],
            #     )
            for key_i in global_crop_1_data.keys():
                # go to cpu and numpy
                global_crop_1_data[key_i] = global_crop_1_data[key_i].cpu().numpy()
            save_ply(global_crop_1_data, f"{data_save_path}/global_crop_1_mask.ply")

            raise ValueError("stop here")

        # prepare mask feature
        if self.enable_mae_loss:
            # we can set local crop to 0
            # assert self.do_mask # only enable mae loss when ibot is enabled
            with torch.no_grad():
                student_patch_feat_global_0_mask_gt = global_crop0_data_dict["feat"][
                    global_mask_0
                ].clone()
                student_patch_feat_global_1_mask_gt = global_crop1_data_dict["feat"][
                    global_mask_1
                ].clone()
                if self.using_coord:
                    student_patch_feat_global_0_mask_gt = (
                        student_patch_feat_global_0_mask_gt[:, 3:]
                    )
                    student_patch_feat_global_1_mask_gt = (
                        student_patch_feat_global_1_mask_gt[:, 3:]
                    )

        # get teacher output:
        if self.do_ema:
            # only global ibot
            (
                teacher_dino_feat_global_pool,
                teacher_patch_feat_global_0_mask,
                teacher_patch_feat_global_1_mask,
            ) = get_teacher_output(
                global_crop0_data_dict,
                global_crop1_data_dict,
                global_mask_0,
                global_mask_1,
            )

            # print("teacher_dino_feat_global_pool dtype", teacher_dino_feat_global_pool.dtype, teacher_dino_feat_global_pool.shape) # torch.float32
            # print("teacher_patch_feat_global_0_mask dtype", teacher_patch_feat_global_0_mask.dtype, teacher_patch_feat_global_0_mask.shape) # torch.float32

            # print("student global fearture operation")
            (
                student_dino_feat_global_pool,
                student_patch_feat_global_0_mask,
                student_patch_feat_global_1_mask,
                student_dino_feat_local_pool_list,
            ) = get_student_output(
                global_crop0_data_dict,
                global_crop1_data_dict,
                local_crop_data_dict_list,
                global_mask_0,
                global_mask_1,
            )

            # print("student_dino_feat_global_pool dtype", student_dino_feat_global_pool.dtype, student_dino_feat_global_pool.shape)
            # print("student_patch_feat_global_0_mask dtype", student_patch_feat_global_0_mask.dtype, student_patch_feat_global_0_mask.shape)

        batch_size = student_dino_feat_global_pool.shape[0]

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Calculate loss
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        loss_dict = {}
        output_dict = {}
        # loss = torch.nn.functional.mse_loss(student_dino_feat_global_pool, teacher_dino_feat_global_pool.detach()).mean()
        # loss_dict["sim_dino_crops_loss"] = loss
        # student_dino_feat_local_pool_list = torch.stack(student_dino_feat_local_pool_list, dim=1) # Batch x Local x Feat_dim
        # loss_dict["sim_dino_crops_loss"] = torch.nn.functional.mse_loss(student_dino_feat_local_pool_list, torch.zeros_like(student_dino_feat_local_pool_list)).mean()

        dino_loss_dict = self.calculate_sim_dino_loss_pool(
            teacher_dino_feat_global_pool,
            student_dino_feat_global_pool,
            student_dino_feat_local_pool_list,
        )

        loss_dict.update(dino_loss_dict)
        # teacher_collapse_indicator = torch.nn.functional.cosine_similarity(teacher_dino_feat_global_pool[:,0].unsqueeze(1), teacher_dino_feat_global_pool[:,0].unsqueeze(0), dim=-1)
        # teacher_collapse_indicator = (teacher_collapse_indicator.sum() - batch_size) / (batch_size * (batch_size - 1))
        # student_collapse_indicator = torch.nn.functional.cosine_similarity(student_dino_feat_global_pool[:,0].unsqueeze(1), student_dino_feat_global_pool[:,0].unsqueeze(0), dim=-1)
        # student_collapse_indicator = (student_collapse_indicator.sum() - batch_size) / (batch_size * (batch_size - 1))

        # # del pool feature to save memory
        # del teacher_dino_feat_global_pool
        # del student_dino_feat_global_pool
        # del student_dino_feat_local_pool_list

        if self.do_ibot:
            ibot_loss_dict = self.calculate_sim_dino_loss_patch(
                teacher_patch_feat_global_0_mask,
                teacher_patch_feat_global_1_mask,
                student_patch_feat_global_0_mask,
                student_patch_feat_global_1_mask,
                teacher_temp=teacher_temp,
                global_mask_weight_0=global_mask_0_weight,
                global_mask_weight_1=global_mask_1_weight,
            )
            teacher_ibot_feat_batch0_crop0_mask0 = teacher_patch_feat_global_0_mask[
                0, :
            ]  # 1x64
            teacher_ibot_feat_batch1_crop0_mask0 = teacher_patch_feat_global_0_mask[
                -1, :
            ]  # use last masked point so from different batch # 1x64
            teacher_ibot_indicator = torch.nn.functional.cosine_similarity(
                teacher_ibot_feat_batch0_crop0_mask0,
                teacher_ibot_feat_batch1_crop0_mask0,
                dim=-1,
            )
            teacher_ibot_indicator = teacher_ibot_indicator.mean()
            output_dict["teacher_ibot_global_indicator"] = teacher_ibot_indicator

            loss_dict.update(ibot_loss_dict)

        if self.enable_mae_loss:
            # calculate mae loss
            # teacher_patch_feat_global_0_mask = self.mae_head(teacher_patch_feat_global_0_mask)
            # teacher_patch_feat_global_1_mask = self.mae_head(teacher_patch_feat_global_1_mask) teacher loss is not updated
            student_patch_feat_global_0_mask = self.mae_head(
                student_patch_feat_global_0_mask
            )
            # student_patch_feat_global_1_mask = self.mae_head(student_patch_feat_global_1_mask)

            mae_loss = nn.functional.mse_loss(
                student_patch_feat_global_0_mask,
                student_patch_feat_global_0_mask_gt.detach(),
            )  # only the global transform 0 is weak transformation
            # + nn.functional.mse_loss(student_patch_feat_global_1_mask, student_patch_feat_global_1_mask_gt.detach())

            loss_dict["global_mae_loss"] = mae_loss

        # del patch feature to save memory
        # del teacher_patch_feat_global_0_mask
        # del teacher_patch_feat_global_1_mask
        # del student_patch_feat_global_0_mask
        # del student_patch_feat_global_1_mask

        if self.training:
            # combine all loss
            loss = 0.0
            loss += self.dino_weight * loss_dict["sim_dino_crops_loss"]
            # output_dict["teacher_collapse_indicator"] = teacher_collapse_indicator
            # output_dict["student_collapse_indicator"] = student_collapse_indicator
            if self.do_ibot:
                loss += self.ibot_weight * loss_dict["sim_ibot_patch_loss"]
                output_dict["teacher_ibot_global_indicator"] = teacher_ibot_indicator

            if self.enable_mae_loss:
                loss += self.mae_weight * loss_dict["global_mae_loss"]

            # assert no NaN in loss
            if torch.isnan(loss):
                raise ValueError("loss is NaN")

            output_dict["loss"] = loss
            output_dict.update(loss_dict)

            return output_dict
        else:
            raise ValueError("Not implemented yet")


from plyfile import PlyData, PlyElement


def save_ply(data_dict, file_path, max_sh_degree=3):
    """
    Save a 3dgs ply file.
      - f_rest channels (extra SH coefficients) are set to zero.
      - Before saving, the opacity and scale values are converted back to their raw forms,
        i.e. the inverse of the sigmoid (logit) and the inverse of exp (log) respectively.
      - The attributes are ordered as:
          x, y, z, nx, ny, nz,
          f_dc_0, f_dc_1, ..., f_dc_{C-1},
          f_rest_0, ..., f_rest_{R-1},
          opacity,
          scale_0, scale_1, ..., scale_{S-1},
          rot_0, rot_1, ..., rot_{Q-1}
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    N = data_dict["coord"].shape[0]

    # Coordinates and normals
    xyz = data_dict["coord"]  # (N, 3)
    normals = data_dict.get("normal", np.zeros_like(xyz))  # (N, 3)

    # f_dc channels from "color" (assumed shape: (N, 3))
    f_dc = data_dict["color"]
    # change rgb to dc
    C0 = 0.28209479177387814
    f_dc = (f_dc / 255.0 - 0.5) / C0
    print("f_dc range", f_dc.min(), f_dc.max())
    num_f_dc = f_dc.shape[1]

    # f_rest channels: set all to zero
    num_f_rest = 3 * (((max_sh_degree + 1) ** 2) - 1)  # For max_sh_degree=3, equals 45
    f_rest = np.zeros((N, num_f_rest), dtype=np.float32)

    # Inverse transform opacity:
    # data_dict["opacity"] was obtained via sigmoid: opacity = 1 / (1 + exp(-raw_opacity))
    # Inverse (logit): raw_opacity = ln(opacity/(1-opacity))
    opacity = data_dict["opacity"]
    if opacity.ndim == 1:
        opacity = opacity.reshape(-1, 1)
    eps = 1e-7  # to prevent division by zero
    opacity = np.clip(opacity, eps, 1 - eps)
    raw_opacity = np.log(opacity / (1 - opacity))

    # Inverse transform scales:
    # data_dict["scale"] was obtained via: scale = exp(raw_scale)
    # So, raw_scale = log(scale)
    scales = data_dict["scale"]
    raw_scales = np.log(scales)
    num_scale = scales.shape[1]

    # Rotation channels (quaternions)
    quat = data_dict["quat"]
    num_quat = quat.shape[1]

    # Build dtype list following the attribute order
    dtype_list = []
    # Coordinates and normals
    for attr in ["x", "y", "z", "nx", "ny", "nz"]:
        dtype_list.append((attr, "f4"))
    # f_dc channels
    for i in range(num_f_dc):
        dtype_list.append((f"f_dc_{i}", "f4"))
    # f_rest channels
    for i in range(num_f_rest):
        dtype_list.append((f"f_rest_{i}", "f4"))
    # Opacity (raw value)
    dtype_list.append(("opacity", "f4"))
    # Scale channels (raw values)
    for i in range(num_scale):
        dtype_list.append((f"scale_{i}", "f4"))
    # Rotation channels (quaternions)
    for i in range(num_quat):
        dtype_list.append((f"rot_{i}", "f4"))

    # Create and fill the structured array
    vertex_all = np.empty(N, dtype=dtype_list)
    vertex_all["x"] = xyz[:, 0]
    vertex_all["y"] = xyz[:, 1]
    vertex_all["z"] = xyz[:, 2]
    vertex_all["nx"] = normals[:, 0]
    vertex_all["ny"] = normals[:, 1]
    vertex_all["nz"] = normals[:, 2]
    for i in range(num_f_dc):
        vertex_all[f"f_dc_{i}"] = f_dc[:, i]
    for i in range(num_f_rest):
        vertex_all[f"f_rest_{i}"] = f_rest[:, i]
    vertex_all["opacity"] = raw_opacity[:, 0]
    for i in range(num_scale):
        vertex_all[f"scale_{i}"] = raw_scales[:, i]
    for i in range(num_quat):
        vertex_all[f"rot_{i}"] = quat[:, i]

    el = PlyElement.describe(vertex_all, "vertex")
    PlyData([el]).write(file_path)
    print(f"Saved {file_path}")
