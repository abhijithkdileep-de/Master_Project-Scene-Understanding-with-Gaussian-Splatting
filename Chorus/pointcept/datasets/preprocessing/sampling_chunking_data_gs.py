import os
import argparse
import numpy as np
import multiprocessing as mp

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path


def _axis_starts(extent, stride, size):
    # extent is the span along an axis (here max_xyz[d], since coord was recentered)
    # If extent <= size -> arange(0, stride, stride) -> [0]
    stop = max(0.0, float(extent) - float(size)) + float(stride)
    return np.arange(0.0, stop, float(stride))

def _l2_normalize_inplace(A: np.ndarray, target_bytes=512*1024*1024, eps=1e-6):
    """
    In-place L2 row-normalization for a huge (K, D) array 'A' (e.g., float16).
    - Streams in blocks using a reusable float32 buffer to minimize allocations.
    - Writes back to A’s dtype in-place.
    """
    assert A.ndim == 2
    K, D = A.shape
    # rows per block such that float32 buffer ~ target_bytes
    rows_per_block = max(1, target_bytes // (D * 4))
    # Reusable float32 buffer
    buf = np.empty((rows_per_block, D), dtype=np.float32)

    for start in range(0, K, rows_per_block):
        end = min(K, start + rows_per_block)
        n = end - start

        # copy A[start:end] -> buf[:n] in float32
        # (single contiguous copy; much cheaper than astype on the whole matrix)
        np.multiply(A[start:end], 1.0, out=buf[:n], casting="unsafe")  # fp16->fp32 copy

        # row norms (float32), einsum is often a hair faster than (buf**2).sum(axis=1)
        ss = np.einsum('ij,ij->i', buf[:n], buf[:n], optimize=True)
        inv = 1.0 / np.maximum(np.sqrt(ss, dtype=np.float32), eps)

        # scale in-place and write back to A’s dtype
        buf[:n] *= inv[:, None]
        np.nan_to_num(buf[:n], copy=False, nan=0.0, posinf=10_000, neginf=-10_000)
        np.clip(buf[:n], -10_000, 10_000, out=buf[:n])
        A[start:end] = buf[:n].astype(A.dtype, copy=False)


def chunking_scene(
    name,
    dataset_root,
    output_dir,
    split,
    grid_size=None,
    chunk_range=(6, 6, 6),
    chunk_stride=(4, 4, 5),
    chunk_minimum_size=100000,
    max_chunk_num=None,
    chunk_z=False,
    return_num_chunks=False,
    debug=False,
):
    if debug:
        print("=============================")
        print("DEBUG MODE, turn off to save chunks")
    print(f"[{name}] Chunking scene in {split} split")
    dataset_root = Path(dataset_root)
    scene_path = dataset_root / split / name
    assets = os.listdir(scene_path)
    data_dict = dict()

    # Load arrays, but skip pc_* files; skip huge features entirely when counting only
    for asset in assets:
        if not asset.endswith(".npy"):
            continue
        if asset.startswith("pc_"):
            continue
        if return_num_chunks and asset in {
            "dino_feat.npy", "dino_feat_index.npy",
            "pe_feat.npy", "pe_feat_index.npy",
            "lang_feat.npy", "lang_feat_index.npy"
        }:
            continue
        key = asset[:-4]
        data_dict[key] = np.load(scene_path / asset)
    if debug:
        print(f"[{name}] Loaded keys: {list(data_dict.keys())}")

    # recenter the coordinates
    coord = data_dict["coord"] - data_dict["coord"].min(axis=0)

    # ------------------------------------------------------------------
    # Treat BIG features as compact if index is present; DO NOT expand.
    # Build O(1) membership sets and orig->compact position maps.
    # ------------------------------------------------------------------
    # DINO
    dino_compact_scene = False
    dino_index_set, dino_pos = None, None
    if "dino_feat" in data_dict:
        if debug:
            print(f"[{name}] DINO feature shape: {data_dict['dino_feat'].shape}")
        if "dino_feat_index" in data_dict:
            dino_compact_scene = True
            dino_idx = data_dict["dino_feat_index"].astype(np.int64)
            dino_index_set = set(dino_idx.tolist())
            dino_pos = {int(ii): i for i, ii in enumerate(dino_idx)}
            # keep data_dict["dino_feat"] as COMPACT (K, D)
        else:
            # dense (N, D) aligned to original indices (0..N-1)
            pass
        if debug:
            print(f"[{name}] DINO compact scene: {dino_compact_scene}")

    # PE-Spatial
    pe_compact_scene = False
    pe_index_set, pe_pos = None, None
    if "pe_feat" in data_dict:
        if debug:
            print(f"[{name}] PE feature shape: {data_dict['pe_feat'].shape}")
        if "pe_feat_index" in data_dict:
            pe_compact_scene = True
            pe_idx = data_dict["pe_feat_index"].astype(np.int64)
            pe_index_set = set(pe_idx.tolist())
            pe_pos = {int(ii): i for i, ii in enumerate(pe_idx)}
        if debug:
            print(f"[{name}] PE compact scene: {pe_compact_scene}")

    # LANG
    lang_compact_scene = False
    lang_index_set, lang_pos = None, None
    if "lang_feat" in data_dict:
        if debug:
            print(f"[{name}] LANG feature shape: {data_dict['lang_feat'].shape}")
        if "lang_feat_index" in data_dict:
            lang_compact_scene = True
            lang_idx = data_dict["lang_feat_index"].astype(np.int64)
            lang_index_set = set(lang_idx.tolist())
            lang_pos = {int(ii): i for i, ii in enumerate(lang_idx)}
            _l2_normalize_inplace(data_dict["lang_feat"])
        else:
            # dense (N, D) aligned to original indices; we won't expand/shrink at grid step
            pass
        if debug:
            print(f"[{name}] LANG compact scene: {lang_compact_scene}")

    # ------------------------------------------------------------------
    # Optional uniform grid decimation (avoid slicing BIG arrays here).
    # Only slice coord and "small" arrays. Keep mapping back to originals.
    # Priority: DINO>PE>LANG>first
    # ------------------------------------------------------------------
    selected_idx_orig = None
    if grid_size is not None:
        print(f"[{name}] Starting grid sampling with size {grid_size}...")
        grid_coord = np.floor(coord / grid_size).astype(int)
        grid_to_indices = {}
        for i, cell in enumerate(grid_coord):
            grid_to_indices.setdefault(tuple(cell), []).append(i)

        has_vf_mask = "valid_feat_mask" in data_dict and isinstance(data_dict["valid_feat_mask"], np.ndarray)

        selected_idx = []
        for indices in grid_to_indices.values():
            choice = None

            # 1) prefer indices with valid DINO (compact) if available
            if dino_index_set is not None:
                dino_valid = [i for i in indices if i in dino_index_set]
                if dino_valid:
                    choice = np.random.choice(dino_valid)

            # 2) otherwise prefer indices with PE-Spatial if available
            if choice is None and pe_index_set is not None:
                pe_valid = [i for i in indices if i in pe_index_set]
                if pe_valid:
                    choice = np.random.choice(pe_valid)

            # 3) otherwise prefer lang-valid rows
            if choice is None:
                lang_valid = []
                if has_vf_mask:
                    lang_valid = [i for i in indices if data_dict["valid_feat_mask"][i] == 1]
                elif lang_index_set is not None:
                    lang_valid = [i for i in indices if i in lang_index_set]
                if lang_valid:
                    choice = np.random.choice(lang_valid)

            # 4) fallback
            if choice is None:
                choice = indices[0]

            selected_idx.append(int(choice))

        selected_idx = np.asarray(selected_idx, dtype=np.int64)

        # Only shrink coord (needed for spatial ops)
        coord = coord[selected_idx]
        # Keep mapping from decimated coords back to original indices
        selected_idx_orig = selected_idx
        data_dict["_selected_idx_orig"] = selected_idx_orig

        # Slice only "small" arrays that align 1:1 with points; avoid BIG feature arrays
        BIG_KEYS = {"dino_feat", "pe_feat", "lang_feat"}
        INDEX_KEYS = {"dino_feat_index", "pe_feat_index", "lang_feat_index"}
        for key in list(data_dict.keys()):
            if key in BIG_KEYS or key in INDEX_KEYS or key == "_selected_idx_orig":
                continue
            arr = data_dict[key]
            if isinstance(arr, np.ndarray) and arr.ndim >= 1 and len(arr) == len(grid_coord):
                data_dict[key] = arr[selected_idx]
        print(f"[{name}] Grid sampling finished with size {grid_size}: {len(grid_to_indices)} -> {len(selected_idx)} points")

    # Range extents
    max_xyz = coord.max(axis=0)
    min_xyz = coord.min(axis=0)  # should be ~0 due to recentering
    if debug:
        print(f"[{name}] xyz_range_max = {max_xyz}, xyz_range_min = {min_xyz}")

    # Build chunk start grid
    if chunk_z:
        xs = _axis_starts(max_xyz[0], chunk_stride[0], chunk_range[0])
        ys = _axis_starts(max_xyz[1], chunk_stride[1], chunk_range[1])
        zs = _axis_starts(max_xyz[2], chunk_stride[2], chunk_range[2])
        if xs.size == 0 or ys.size == 0 or zs.size == 0:
            chunks = np.zeros((0, 3), dtype=float)
        else:
            xg, yg, zg = np.meshgrid(xs, ys, zs, indexing="ij")
            chunks = np.stack([xg.reshape(-1), yg.reshape(-1), zg.reshape(-1)], axis=-1)
    else:
        xs = _axis_starts(max_xyz[0], chunk_stride[0], chunk_range[0])
        ys = _axis_starts(max_xyz[1], chunk_stride[1], chunk_range[1])
        if xs.size == 0 or ys.size == 0:
            chunks = np.zeros((0, 2), dtype=float)
        else:
            xg, yg = np.meshgrid(xs, ys, indexing="ij")
            chunks = np.concatenate([xg.reshape([-1, 1]), yg.reshape([-1, 1])], axis=-1)

    # Helper: build mask for a given chunk start (2D or 3D) over *current coord*
    def chunk_mask(start):
        if chunk_z:
            return (
                (coord[:, 0] >= start[0]) & (coord[:, 0] < start[0] + chunk_range[0]) &
                (coord[:, 1] >= start[1]) & (coord[:, 1] < start[1] + chunk_range[1]) &
                (coord[:, 2] >= start[2]) & (coord[:, 2] < start[2] + chunk_range[2])
            )
        else:
            return (
                (coord[:, 0] >= start[0]) & (coord[:, 0] < start[0] + chunk_range[0]) &
                (coord[:, 1] >= start[1]) & (coord[:, 1] < start[1] + chunk_range[1])
            )

    def _count_nonzero_rows(A):
        if A.size == 0:
            return 0
        return int(np.any(A != 0, axis=1).sum())
    
    # -------------------------------
    # Prefilter by chunk_minimum_size
    # -------------------------------
    chunks = np.asarray(chunks)            # shape (M, 2) or (M, 3)
    valid_idx = []
    prefilter_point_counts = []            # cache counts to avoid recompute later when selecting by points
    num_before = len(chunks)
    for i, ch in enumerate(chunks):
        m = chunk_mask(ch)
        cnt = int(m.sum())
        if cnt >= chunk_minimum_size:
            valid_idx.append(i)
            prefilter_point_counts.append(cnt)
        elif debug:
            print(f"[{name}] chunk {i} too small with {cnt} 3dgs, skipping")
    print(f"[{name}] {len(valid_idx)}/{num_before} chunks passed minimum size {chunk_minimum_size}")

    # Keep only valid chunks; keep counts aligned
    if len(valid_idx) == 0:
        chunks = chunks[:0]                # empty array of correct shape
        prefilter_point_counts = np.asarray([], dtype=np.int32)
    else:
        valid_idx = np.asarray(valid_idx, dtype=np.intp)
        chunks = chunks[valid_idx]
        prefilter_point_counts = np.asarray(prefilter_point_counts, dtype=np.int32)

    # ------------------------------
    # Selection if too many chunks 
    # ------------------------------
    if max_chunk_num is not None and len(chunks) > max_chunk_num:
        use_segment = "segment" in data_dict
        seg = data_dict.get("segment", None)

        if use_segment:
            scores = np.empty(len(chunks), dtype=np.int32)
            for i, ch in enumerate(chunks):
                m = chunk_mask(ch)
                scores[i] = int((seg[m] != -1).sum())
        else:
            # Score by points, already computed counts during prefilter
            scores = prefilter_point_counts

        n = scores.size
        k = int(min(max_chunk_num, n))
        # Top-k via argpartition, then sort those k by score desc
        topk_unsorted = np.argpartition(scores, -k)[-k:]
        topk_sorted = topk_unsorted[np.argsort(scores[topk_unsorted])[::-1]]
        idx = topk_sorted.astype(np.intp, copy=False)

        chunks = chunks[idx]
        # Keep counts aligned too (only used if !use_segment)
        if not use_segment:
            prefilter_point_counts = scores[idx]
        print(f"[{name}] selected {len(chunks)} chunks with most {'valid segments' if use_segment else 'points'}")

    # Create output split name
    def make_split_name():
        if grid_size is not None:
            if chunk_z:
                return (f"{split}_grid{grid_size * 100:.1f}cm_"
                        f"chunk{chunk_range[0]}x{chunk_range[1]}x{chunk_range[2]}_"
                        f"stride{chunk_stride[0]}x{chunk_stride[1]}x{chunk_stride[2]}")
            else:
                return (f"{split}_grid{grid_size * 100:.1f}cm_"
                        f"chunk{chunk_range[0]}x{chunk_range[1]}_"
                        f"stride{chunk_stride[0]}x{chunk_stride[1]}")
        else:
            if chunk_z:
                return (f"{split}_"
                        f"chunk{chunk_range[0]}x{chunk_range[1]}x{chunk_range[2]}_"
                        f"stride{chunk_stride[0]}x{chunk_stride[1]}x{chunk_stride[2]}")
            else:
                return (f"{split}_"
                        f"chunk{chunk_range[0]}x{chunk_range[1]}_"
                        f"stride{chunk_stride[0]}x{chunk_stride[1]}")

    chunk_split_name = make_split_name()

    # Iterate and either count or write
    chunk_idx = 0
    for idx, ch in enumerate(chunks):
        if debug:
            print(f"[{name}] chunk {ch} chunk_range {chunk_range}")
        mask = chunk_mask(ch)

        chunk_data_name = f"{name}_{chunk_idx}"
        if np.sum(mask) < chunk_minimum_size:
            print(f"[{name}] chunk {idx} too small with {np.sum(mask)} 3dgs, skipping")
            continue

        # Map decimated mask back to ORIGINAL indices when needed
        if selected_idx_orig is not None:
            orig_idx = selected_idx_orig[mask]  # shape (M,)
        else:
            # no grid decimation path: coord already aligns to originals
            orig_idx = np.nonzero(mask)[0]

        # ----- Skip if chunk has zero valid lang or zero valid dino -----
        # LANG check (prefer valid_feat_mask if present; otherwise lang_index_set)
        has_lang_rows = True
        if "lang_feat" in data_dict:
            vf_global = data_dict.get("valid_feat_mask", None)
            if isinstance(vf_global, np.ndarray):
                if len(vf_global) == len(coord):
                    has_lang_rows = (vf_global[mask].sum() > 0)
                else:
                    has_lang_rows = (vf_global[orig_idx].sum() > 0)
            elif lang_index_set is not None:
                has_lang_rows = any(int(i) in lang_index_set for i in orig_idx)
            else:
                # dense lang without mask: treat nonzero rows as valid
                LF = data_dict["lang_feat"]
                if len(LF) == len(coord):
                    has_lang_rows = (_count_nonzero_rows(LF[mask]) > 0)
                else:
                    has_lang_rows = (_count_nonzero_rows(LF[orig_idx]) > 0)

        if not has_lang_rows:
            print(f"[{name}] chunk {idx}: 0 valid lang rows; skip")
            continue

        # DINO check
        has_dino_rows = True
        if "dino_feat" in data_dict:
            if dino_compact_scene:
                has_dino_rows = any(int(i) in dino_index_set for i in orig_idx)
            else:
                DF = data_dict["dino_feat"]
                if len(DF) == len(coord):
                    has_dino_rows = (_count_nonzero_rows(DF[mask]) > 0)
                else:
                    has_dino_rows = (_count_nonzero_rows(DF[orig_idx]) > 0)

        if not has_dino_rows:
            print(f"[{name}] chunk {idx}: 0 valid dino rows; skip")
            continue

        # PE-Spatial check
        has_pe_rows = True
        if "pe_feat" in data_dict:
            if pe_compact_scene:
                has_pe_rows = any(int(i) in pe_index_set for i in orig_idx)
            else:
                PF = data_dict["pe_feat"]
                if len(PF) == len(coord):
                    has_pe_rows = (_count_nonzero_rows(PF[mask]) > 0)
                else:
                    has_pe_rows = (_count_nonzero_rows(PF[orig_idx]) > 0)

        if not has_pe_rows:
            print(f"[{name}] chunk {idx}: 0 valid pe rows; skip")
            continue

        if return_num_chunks or debug:
            chunk_idx += 1
            continue

        # ---------------- Save ----------------
        if output_dir is not None:
            chunk_save_path = Path(output_dir) / chunk_split_name / chunk_data_name
        else:
            chunk_save_path = dataset_root / chunk_split_name / chunk_data_name
        chunk_save_path.mkdir(parents=True, exist_ok=True)

        # Save "small" keys directly; avoid saving BIG features here
        SKIP_KEYS = {
            "lang_feat", "valid_feat_mask", "lang_feat_index",
            "dino_feat", "dino_feat_index",
            "pe_feat", "pe_feat_index", "_selected_idx_orig"
        }
        for key in list(data_dict.keys()):
            if key in SKIP_KEYS:
                continue
            arr = data_dict[key]
            if not isinstance(arr, np.ndarray):
                continue
            if len(arr) == len(coord):
                np.save(chunk_save_path / f"{key}.npy", arr[mask])
            elif arr.shape[0] >= orig_idx.max() + 1:  # aligns to originals
                np.save(chunk_save_path / f"{key}.npy", arr[orig_idx])
            else:
                # size-mismatch arrays are ignored
                pass

        # ----- Save language compact per chunk -----
        if "lang_feat" in data_dict:
            # Build chunk_valid_mask (aligned to chunk, length = sum(mask))
            vf_global = data_dict.get("valid_feat_mask", None)
            if isinstance(vf_global, np.ndarray):
                if len(vf_global) == len(coord):
                    chunk_valid_mask = vf_global[mask].astype(np.int32)
                else:
                    chunk_valid_mask = vf_global[orig_idx].astype(np.int32)
            else:
                # No valid_feat_mask available. Derive from presence in lang_index_set or from nonzero rows
                if lang_index_set is not None:
                    chunk_valid_mask = np.array([1 if int(i) in lang_index_set else 0 for i in orig_idx], dtype=np.int32)
                else:
                    # Dense without mask: consider nonzero rows valid
                    LF = data_dict["lang_feat"]
                    if len(LF) == len(coord):
                        sub = LF[mask]
                    else:
                        sub = LF[orig_idx]
                    nz = np.any(sub != 0, axis=1)
                    chunk_valid_mask = nz.astype(np.int32)

            # Build lang_feat_index + lang_feat for the chunk
            valid_local = np.flatnonzero(chunk_valid_mask).astype(np.int32)
            if lang_compact_scene:
                if valid_local.size > 0:
                    # gather from compact using lang_pos
                    compact_rows = [lang_pos[int(orig_idx[k])] for k in valid_local if int(orig_idx[k]) in lang_pos]
                    if len(compact_rows) > 0:
                        lang_kept = data_dict["lang_feat"][np.asarray(compact_rows, dtype=np.int64)]
                    else:
                        lang_kept = np.zeros((0, data_dict["lang_feat"].shape[1]), dtype=data_dict["lang_feat"].dtype)
                else:
                    lang_kept = np.zeros((0, data_dict["lang_feat"].shape[1]), dtype=data_dict["lang_feat"].dtype)
                np.save(chunk_save_path / "valid_feat_mask.npy", chunk_valid_mask)
                np.save(chunk_save_path / "lang_feat_index.npy", valid_local)
                np.save(chunk_save_path / "lang_feat.npy", lang_kept)
            else:
                # Dense lang: slice then compact-save
                LF = data_dict["lang_feat"]
                if len(LF) == len(coord):
                    chunk_lang_full = LF[mask]
                else:
                    chunk_lang_full = LF[orig_idx]
                if valid_local.size > 0:
                    chunk_lang_valid = chunk_lang_full[valid_local]
                else:
                    chunk_lang_valid = np.zeros((0, chunk_lang_full.shape[1]), dtype=chunk_lang_full.dtype)
                np.save(chunk_save_path / "valid_feat_mask.npy", chunk_valid_mask)
                np.save(chunk_save_path / "lang_feat_index.npy", valid_local)
                np.save(chunk_save_path / "lang_feat.npy", chunk_lang_valid)

        # ----- Save DINO (compact if scene was compact) -----
        if "dino_feat" in data_dict:
            if dino_compact_scene:
                local_has_dino = [k for k, ii in enumerate(orig_idx.tolist()) if ii in dino_index_set]
                if local_has_dino:
                    compact_rows = [dino_pos[int(orig_idx[k])] for k in local_has_dino]
                    dino_kept = data_dict["dino_feat"][np.asarray(compact_rows, dtype=np.int64)]
                    dino_idx_local = np.asarray(local_has_dino, dtype=np.int32)
                else:
                    dino_kept = np.zeros((0, data_dict["dino_feat"].shape[1]), dtype=data_dict["dino_feat"].dtype)
                    dino_idx_local = np.zeros((0,), dtype=np.int32)
                np.save(chunk_save_path / "dino_feat_index.npy", dino_idx_local)
                np.save(chunk_save_path / "dino_feat.npy", dino_kept)
            else:
                DF = data_dict["dino_feat"]
                if len(DF) == len(coord):
                    chunk_dino_full = DF[mask]
                else:
                    chunk_dino_full = DF[orig_idx]
                np.save(chunk_save_path / "dino_feat.npy", chunk_dino_full)

        # ----- Save PE-Spatial (compact if scene was compact) -----
        if "pe_feat" in data_dict:
            if pe_compact_scene:
                local_has_pe = [k for k, ii in enumerate(orig_idx.tolist()) if ii in pe_index_set]
                if local_has_pe:
                    compact_rows = [pe_pos[int(orig_idx[k])] for k in local_has_pe]
                    pe_kept = data_dict["pe_feat"][np.asarray(compact_rows, dtype=np.int64)]
                    pe_idx_local = np.asarray(local_has_pe, dtype=np.int32)
                else:
                    pe_kept = np.zeros((0, data_dict["pe_feat"].shape[1]), dtype=data_dict["pe_feat"].dtype)
                    pe_idx_local = np.zeros((0,), dtype=np.int32)
                np.save(chunk_save_path / "pe_feat_index.npy", pe_idx_local)
                np.save(chunk_save_path / "pe_feat.npy", pe_kept)
            else:
                PF = data_dict["pe_feat"]
                if len(PF) == len(coord):
                    chunk_pe_full = PF[mask]
                else:
                    chunk_pe_full = PF[orig_idx]
                np.save(chunk_save_path / "pe_feat.npy", chunk_pe_full)

        chunk_idx += 1

    print(f"[{name}] in total {chunk_idx} valid chunks")
    return chunk_idx if return_num_chunks else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the Pointcept processed ScanNet++ dataset.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output root for chunked scenes (default: inside dataset_root).",
    )
    parser.add_argument(
        "--split",
        required=True,
        default="train",
        type=str,
        help="Split to process.",
    )
    parser.add_argument(
        "--grid_size",
        default=None,
        type=float,
        help="Grid size for initial grid sampling",
    )
    parser.add_argument(
        "--chunk_range",
        default=[6, 6, 6],
        type=int,
        nargs="+",
        help="Range of each chunk, e.g. --chunk_range 6 6 6. With --chunk_z must have 3 values.",
    )
    parser.add_argument(
        "--chunk_stride",
        default=[4, 4, 5],
        type=int,
        nargs="+",
        help="Stride of each chunk, e.g. --chunk_stride 4 4 5. With --chunk_z must have 3 values.",
    )
    parser.add_argument(
        "--chunk_minimum_size",
        default=50000,
        type=int,
        help="Minimum number of points in each chunk",
    )
    parser.add_argument(
        "--num_workers",
        default=mp.cpu_count(),
        type=int,
        help="Num workers for preprocessing.",
    )
    parser.add_argument(
        "--subset_list",
        type=str,
        default=None,
        help=("File containing dataset entries (folders) to keep; otherwise use all."),
    )
    parser.add_argument(
        "--max_chunk_num",
        type=int,
        default=None,
        help="Maximum number of chunks to process per scene (selects top by points or valid segments).",
    )
    parser.add_argument("--single_process", action="store_true")
    parser.add_argument("--chunk_z", action="store_true", help="Enable 3D chunking along Z.")
    parser.add_argument(
        "--return_num_chunks",
        action="store_true",
        help="Only count total valid chunks (after filtering/selection) and exit.",
    )
    parser.add_argument("--debug", action="store_true")

    config = parser.parse_args()
    config.dataset_root = Path(config.dataset_root)

    # Validate chunk_range/stride lengths
    cr = list(config.chunk_range)
    cs = list(config.chunk_stride)

    if config.chunk_z:
        if len(cr) != 3 or len(cs) != 3:
            raise ValueError("--chunk_z requires --chunk_range and --chunk_stride to each have exactly 3 values.")
    else:
        # Allow 2 or 3 values; ignore Z if provided
        if len(cr) < 2 or len(cs) < 2:
            raise ValueError("--chunk_range/--chunk_stride must have at least 2 values when --chunk_z is not set.")
        if len(cr) > 2:
            if config.debug:
                print(f"[warn] --chunk_z is off; ignoring extra chunk_range value(s): {cr[2:]}")
            cr = cr[:2]
        if len(cs) > 2:
            if config.debug:
                print(f"[warn] --chunk_z is off; ignoring extra chunk_stride value(s): {cs[2:]}")
            cs = cs[:2]
    config.chunk_range = tuple(cr)
    config.chunk_stride = tuple(cs)

    data_list = os.listdir(config.dataset_root / config.split)
    if config.subset_list is not None:
        subset_path = Path(config.subset_list)
        with subset_path.open("r") as f:
            subset_list = {line.strip() for line in f if line.strip()}
        data_list = [name for name in data_list if name in subset_list]

    print("===== Chunking 3DGS Data =====")
    print(f"Processing {len(data_list)} scenes in {config.split} split")

    if config.return_num_chunks:
        # Count-only path
        if not config.single_process:
            with ProcessPoolExecutor(max_workers=config.num_workers) as pool:
                counts = list(
                    pool.map(
                        chunking_scene,
                        data_list,
                        repeat(config.dataset_root),
                        repeat(config.output_dir),
                        repeat(config.split),
                        repeat(config.grid_size),
                        repeat(config.chunk_range),
                        repeat(config.chunk_stride),
                        repeat(config.chunk_minimum_size),
                        repeat(config.max_chunk_num),
                        repeat(config.chunk_z),
                        repeat(True),  # return_num_chunks
                        repeat(config.debug),
                    )
                )
        else:
            counts = []
            for name in data_list:
                c = chunking_scene(
                    name,
                    config.dataset_root,
                    config.output_dir,
                    config.split,
                    config.grid_size,
                    config.chunk_range,
                    config.chunk_stride,
                    config.chunk_minimum_size,
                    config.max_chunk_num,
                    config.chunk_z,
                    True,  # return_num_chunks
                    config.debug,
                )
                counts.append(c)

        total = int(np.sum([c if c is not None else 0 for c in counts]))
        print(f"[SUMMARY] Total valid chunks across all scenes: {total}")
    else:
        # Normal chunking path (writes to disk)
        if not config.single_process:
            with ProcessPoolExecutor(max_workers=config.num_workers) as pool:
                _ = list(
                    pool.map(
                        chunking_scene,
                        data_list,
                        repeat(config.dataset_root),
                        repeat(config.output_dir),
                        repeat(config.split),
                        repeat(config.grid_size),
                        repeat(config.chunk_range),
                        repeat(config.chunk_stride),
                        repeat(config.chunk_minimum_size),
                        repeat(config.max_chunk_num),
                        repeat(config.chunk_z),
                        repeat(False),  # return_num_chunks
                        repeat(config.debug),
                    )
                )
        else:
            print("Using single process for chunking...")
            for name in data_list:
                chunking_scene(
                    name,
                    config.dataset_root,
                    config.output_dir,
                    config.split,
                    config.grid_size,
                    config.chunk_range,
                    config.chunk_stride,
                    config.chunk_minimum_size,
                    config.max_chunk_num,
                    config.chunk_z,
                    False,  # return_num_chunks
                    config.debug,
                )
        print("All scenes chunked!")
