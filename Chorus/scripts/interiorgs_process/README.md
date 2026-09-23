### InteriorGS 2D Adaptation Preprocessing

These scripts prepare the optional 2D adaptation assets used by
`Interior2DGSDataset`. Run them from `scripts/interiorgs_process` after the
normal Gaussian parameter `*.npy` files already exist.

Required per-scene input layout:

```bash
<preprocessed_root>/<split>/<scene_id>/
├── color.npy
├── coord.npy
├── opacity.npy
├── quat.npy
├── scale.npy
└── segment.npy
```

The raw InteriorGS scenes are passed separately through `--interior_gs_path`.

Install the extra preprocessing/runtime packages in the active environment:

```bash
bash jobs/install.sh
```

Sample candidate cameras, render RGB/depth images, and write the filtered camera
metadata:

```bash
python sample_camera_view_and_create_transform_interiorgs.py \
  --interior_gs_path /path/to/interior_gs/scenes \
  --interior_gs_preprocessed_path /path/to/preprocessed_root
```

This writes `<scene>/render`, `<scene>/render_filtered`,
`transforms_camera_positions_filtered.json`, and
`visiable_gaussian_masks_per_frame_filtered.npy`.

Build expanded per-view visibility boxes:

```bash
python augment_the_visable_points_bbox_interiorgs.py \
  --scene_root_path /path/to/preprocessed_root/<split>
```

This writes `visiable_gaussian_masks_per_frame_filtered_box.npy` and
`visiable_gaussian_masks_per_frame_filtered_box_mask.npy`.

Pair related camera views:

```bash
python pair_the_visable_frames_bbox_interiorgs.py \
  --scene_root_path /path/to/preprocessed_root/<split> \
  --batch_size 4
```

This writes `visiable_gaussian_masks_per_frame_filtered_pair_top4.npy`.

`Interior2DGSDataset` defaults to `render_filtered`, `pair_top_k=4`, and the
existing `visiable_*` file spelling used by these scripts. It falls back to
`render` if `render_filtered` is absent.
