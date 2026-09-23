"""Evaluate input-aligned Chorus language features on 3D OVS PNG masks."""
import argparse
import contextlib
import csv
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tools.ovs_colmap import load_camera_data
from tools.ovs_results import save_query_artifacts, save_comparison_reports
from tools.ovs_qualitative import save_semantic_visualizations
from tools.gaussian_renderer import get_camera_intrinsics, render_semantic_mask

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', type=Path, help='Preprocessed Gaussian .npy folder in COLMAP world coordinates')
    p.add_argument('--feature-path', type=Path, help='Saved Chorus language feature tensor')
    p.add_argument('--dataset-root', required=True, type=Path, help='Extracted 3dovs folder')
    p.add_argument('--prepared-root', required=True, type=Path, help='Undistorted scene folder with images and sparse/0')
    p.add_argument('--scene', required=True)
    p.add_argument('--output-dir', type=Path, help='Output root; writes 3dovs_new_runs/argmax/<scene>')
    p.add_argument('--query', help='Evaluate one label, while still comparing it with every scene label')
    p.add_argument('--check-dataset', action='store_true')
    p.add_argument('--text-model', default='google/siglip2-so400m-patch14-384')
    p.add_argument('--text-embeddings', type=Path, help='Optional .pt dictionary of scene-label text embeddings')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if not args.check_dataset and any(v is None for v in (args.input_root, args.feature_path, args.output_dir)):
        p.error('--input-root, --feature-path and --output-dir are required for evaluation')
    return args


def load_scene(root, feature_path):
    import torch
    scene = {k: np.load(root / (k + '.npy')) for k in ['coord', 'scale', 'quat', 'opacity']}
    n = len(scene['coord'])
    for key, shape in [('coord', (n, 3)), ('scale', (n, 3)), ('quat', (n, 4))]:
        if scene[key].shape != shape or not np.isfinite(scene[key]).all():
            raise ValueError(f'Invalid {key}: expected finite {shape}')
    if n == 0 or np.any(np.linalg.norm(scene['quat'], axis=1) < 1e-8):
        raise ValueError('Geometry must be nonempty with nonzero quaternions')
    scene['opacity'] = scene['opacity'].reshape(-1)
    if scene['opacity'].shape != (n,) or not np.isfinite(scene['opacity']).all():
        raise ValueError('Opacity must have one finite value per Gaussian')
    if np.any(scene['scale'] <= 0) or np.any((scene['opacity'] < 0) | (scene['opacity'] > 1)):
        raise ValueError('Expected physical positive scales and opacity probabilities, as exported by Chorus preprocessing')
    features = torch.load(feature_path, map_location='cpu', weights_only=True)
    sidecar = feature_path.with_name(feature_path.stem + '_index.npy')
    if sidecar.exists():
        indices = np.load(sidecar)
        if indices.ndim != 1 or indices.dtype.kind not in 'iu' or len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= n):
            raise ValueError('Invalid feature index sidecar')
        scene = {k: v[indices] for k, v in scene.items()}
    if not isinstance(features, torch.Tensor):
        details = list(features) if isinstance(features, dict) else type(features).__name__
        raise ValueError(f'{feature_path}: expected a language tensor, got {type(features).__name__}: {details}')
    if features.ndim != 2 or not features.numel() or not features.is_floating_point():
        raise ValueError(f'{feature_path}: expected nonempty floating N x D tensor; shape={tuple(features.shape)}, dtype={features.dtype}')
    if len(features) != len(scene['coord']):
        raise ValueError(
            f'Features are not aligned: {feature_path} has {len(features):,} rows; '
            f'geometry has {len(scene["coord"]):,} rows (original {n:,}); '
            f'index sidecar={sidecar if sidecar.exists() else "absent"}. '
            'Use the exact geometry used for inference and its matching sidecar. Do not truncate rows.'
        )
    bad_values = bad_rows = 0
    first_bad_row = None
    for start in range(0, len(features), 32768):
        invalid = ~torch.isfinite(features[start:start + 32768])
        rows = invalid.any(dim=1)
        bad_values += int(invalid.sum())
        bad_rows += int(rows.sum())
        if first_bad_row is None and rows.any():
            first_bad_row = start + int(rows.nonzero()[0, 0])
    if bad_values:
        raise ValueError(
            f'{feature_path}: {bad_values:,} nonfinite values in {bad_rows:,}/{len(features):,} '
            f'rows; first bad row={first_bad_row}; shape={tuple(features.shape)}, dtype={features.dtype}. '
            'Inspect inference outputs/logs; NaN/Inf features cannot produce valid evaluation scores.'
        )
    # Keep the saved dtype and normalize one chunk at a time to limit memory use.
    return scene, features


from tools.ovs_dataset import load_frames, prediction_in_original_view


def encode_text(labels, args):
    import torch
    if args.text_embeddings:
        cached = torch.load(args.text_embeddings, map_location='cpu', weights_only=True)
        features = torch.stack([torch.as_tensor(cached[label]) for label in labels])
    else:
        import transformers
        config = transformers.AutoConfig.from_pretrained(args.text_model)
        text_config = getattr(config, 'text_config', config)
        model_classes = {
            'siglip_text_model': 'SiglipTextModel',
            'siglip2_text_model': 'Siglip2TextModel',
        }
        if text_config.model_type not in model_classes:
            raise ValueError(f'Unsupported text architecture: {text_config.model_type}')
        model_class = getattr(transformers, model_classes[text_config.model_type])
        print(f'Text encoder: {args.text_model} ({model_class.__name__})', flush=True)
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.text_model)
        model, loading = model_class.from_pretrained(
            args.text_model, config=text_config, output_loading_info=True,
        )
        if loading.get('missing_keys') or loading.get('mismatched_keys') or loading.get('error_msgs'):
            raise ValueError(f'Text weights did not load completely: {loading}')
        model = model.to(args.device).eval()
        tokens = tokenizer(labels, padding='max_length', max_length=64, truncation=True, return_tensors='pt').to(args.device)
        with torch.inference_mode():
            features = model(**tokens).pooler_output.cpu()
    if features.ndim != 2 or not torch.isfinite(features).all():
        raise ValueError('Text embeddings must be finite vectors')
    return torch.nn.functional.normalize(features.float(), dim=-1)


def semantic_similarities(features, embeddings, device='cuda', chunk_size=32768):
    """Cosine similarity of every Gaussian to every scene class."""
    import torch
    if features.ndim != 2 or embeddings.ndim != 2 or features.shape[1] != embeddings.shape[1]:
        raise ValueError('Language/text feature dimensions differ')
    if len(embeddings) == 0:
        raise ValueError('At least one scene label is required')
    embeddings = torch.nn.functional.normalize(embeddings.float(), dim=-1).to(device)
    scores = np.empty((len(features), len(embeddings)), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(features), chunk_size):
            chunk = torch.nn.functional.normalize(
                features[start:start + chunk_size].to(device=device, dtype=torch.float32), dim=-1)
            scores[start:start + len(chunk)] = (chunk @ embeddings.T).cpu().numpy()
    return scores


def semantic_labels(similarities):
    """Assign every Gaussian its highest cosine-similarity scene class."""
    if similarities.ndim != 2 or similarities.shape[1] == 0 or not np.isfinite(similarities).all():
        raise ValueError('Expected finite N x C cosine similarities')
    return similarities.argmax(axis=1).astype(np.int32) + 1


def iou(prediction, target):
    union = np.logical_or(prediction, target).sum()
    return float(np.logical_and(prediction, target).sum() / union) if union else 1.0


def binary_macc(prediction, target):
    """Mean pixel accuracy of foreground and background classes present in GT."""
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    accuracies = [np.mean(prediction[target == label] == label)
                  for label in (False, True) if np.any(target == label)]
    return float(np.mean(accuracies))


def render_semantic_results(labels, frames, scene, all_queries, evaluated_queries):
    """Render all classes and choose one winning class per original-image pixel."""
    geometry = {k: scene[k] for k in ('coord', 'scale', 'quat', 'opacity')}
    selected = [np.flatnonzero(labels == class_id + 1) for class_id in range(len(all_queries))]
    results = {query: [] for query in evaluated_queries}
    query_ids = {query: i + 1 for i, query in enumerate(all_queries)}
    for frame in frames:
        best_soft = np.zeros(frame['shape'], dtype=np.float32)
        pixel_labels = np.zeros(frame['shape'], dtype=np.int32)
        for class_id, indices in enumerate(selected, start=1):
            if not len(indices):
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                _, soft = render_semantic_mask(
                    **geometry, selected_indices=indices, image=frame['image'],
                    camera=frame['camera'], image_shape=frame['render_shape'])
            soft = prediction_in_original_view(soft, frame)
            winner = (soft > 0) & (soft > best_soft)
            pixel_labels[winner] = class_id
            best_soft[winner] = soft[winner]
        for query in evaluated_queries:
            if query in frame['masks']:
                mask = (pixel_labels == query_ids[query]).astype(np.uint8)
                results[query].append((frame, mask, iou(mask, frame['masks'][query])))
    return results


def main():
    args = parse_args()
    frames = load_frames(args.dataset_root, args.scene, args.prepared_root)
    all_queries = sorted({q for frame in frames for q in frame['masks']})
    if args.check_dataset:
        print(json.dumps({'scene': args.scene, 'frames': len(frames), 'queries': all_queries,
                          'mask_count': sum(len(f['masks']) for f in frames)}, indent=2))
        return
    import torch
    if torch.device(args.device).type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Run evaluation inside a GPU allocation or use --device cpu.')
    if not all_queries:
        raise ValueError('Annotations contain no object labels')
    if args.query and args.query not in all_queries:
        raise ValueError(f'Unknown query {args.query!r}; available: {all_queries}')
    evaluated_queries = [args.query] if args.query else all_queries
    scene, features = load_scene(args.input_root, args.feature_path)
    text = encode_text(all_queries, args)
    if features.shape[1] != text.shape[1]:
        raise ValueError(f'Language/text dimensions differ: {features.shape[1]} vs {text.shape[1]}')
    similarities = semantic_similarities(features, text, args.device)
    labels = semantic_labels(similarities)
    output_dir = args.output_dir / '3dovs_new_runs' / 'argmax' / args.scene
    output_dir.mkdir(parents=True, exist_ok=True)
    results_by_query = render_semantic_results(labels, frames, scene, all_queries, all_queries)
    qualitative_manifest = save_semantic_visualizations(
        output_dir, frames, results_by_query, all_queries, args, 'Chorus')
    rows, summaries = [], []
    for qi, query in enumerate(evaluated_queries):
        class_id = all_queries.index(query) + 1
        scores = similarities[:, class_id - 1]
        selection = (labels == class_id).astype(np.uint8)
        query_results = results_by_query[query]
        mean = float(np.mean([result[2] for result in query_results]))
        mean_macc = float(np.mean([
            binary_macc(mask, frame['masks'][query])
            for frame, mask, _ in query_results]))
        summary = dict(query=query, class_index=class_id - 1,
                       class_vocabulary=all_queries, mean_iou=mean,
                       mAcc=mean_macc,
                       selected_gaussians=int(selection.sum()),
                       raw_selected_gaussians=int(selection.sum()),
                       frames=len(query_results),
                       score_quantiles=dict(zip(['min', 'p50', 'p90', 'p99', 'max'],
                           map(float, np.quantile(scores, [0, .5, .9, .99, 1])))))
        summary['artifacts'] = save_query_artifacts(
            output_dir, query, evaluated_queries, scores, selection, summary,
            query_results, scene, args)
        summaries.append(summary)
        for fi, (frame, mask, score) in enumerate(query_results):
            filename = f'q{qi:03d}_f{fi:03d}.png'
            Image.fromarray(mask * 255).save(output_dir / filename)
            rows.append(dict(query=query, image=frame['name'], iou=score,
                             mAcc=binary_macc(mask, frame['masks'][query]),
                             mask=filename,
                             predicted_pixels=int(mask.sum()),
                             gt_pixels=int(frame['masks'][query].sum()),
                             intersection_pixels=int(np.logical_and(mask, frame['masks'][query]).sum())))
        print(f'{query}: mean IoU={mean:.4f}, selected={int(selection.sum()):,}', flush=True)
    report = dict(scene=args.scene, dataset='3dovs', protocol='cosine_argmax_segmentation',
                  class_vocabulary=all_queries,
                  qualitative_manifest=qualitative_manifest,
                  score_type='cosine_similarity_to_scene_labels',
                  mask_coordinate_space='original_distorted_images',
                  geometry_coordinate_space='COLMAP_world', text_prompt='{label}',
                  voting='none', renderer='selected_gaussian_footprints_pixel_argmax',
                  query_macro_miou=float(np.mean([s['mean_iou'] for s in summaries])),
                  query_macro_macc=float(np.mean([s['mAcc'] for s in summaries])),
                  frame_query_miou=float(np.mean([r['iou'] for r in rows])),
                  frame_query_macc=float(np.mean([r['mAcc'] for r in rows])),
                  queries=summaries,
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    save_comparison_reports(output_dir, report)
    (output_dir / 'summary.json').write_text(json.dumps(report, indent=2))
    with (output_dir / 'per_frame.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Cosine argmax: query mIoU={report["query_macro_miou"]:.4f}; saved {output_dir}', flush=True)


if __name__ == '__main__':
    main()
