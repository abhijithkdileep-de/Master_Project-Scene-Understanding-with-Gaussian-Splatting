"""Evaluate input-aligned Chorus language features on LERF polygon annotations."""
import argparse
import contextlib
import csv
import io
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tools.lerf_colmap import load_camera_data
from tools.lerf_results import save_query_artifacts, save_comparison_reports, query_folder, safe_name
from tools.lerf_coloured_masks import save_coloured_mask_results
from tools.gaussian_renderer import get_camera_intrinsics, render_semantic_mask

CANONICAL = ['object', 'things', 'stuff', 'texture']
DEFAULT_THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.80)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', required=True, type=Path, help='Preprocessed Gaussian .npy folder in COLMAP world coordinates')
    p.add_argument('--feature-path', required=True, type=Path, help='Saved Chorus language tensor, e.g. outputs/teatime/teatime_feat.pt')
    p.add_argument('--dataset-root', required=True, type=Path, help='Extracted lerf_ovs folder')
    p.add_argument('--scene', required=True)
    p.add_argument('--output-dir', required=True, type=Path, help='Output root; writes lerf_new_runs/results_XX/<scene>')
    p.add_argument('--query')
    p.add_argument('--thresholds', type=float, nargs='+', default=DEFAULT_THRESHOLDS,
                   help='Fixed relevance thresholds applied to every query (default: 0.20 0.30 0.40 0.50 0.60 0.80)')
    p.add_argument('--temperature', type=float, default=10.0)
    p.add_argument('--text-model', default='google/siglip2-so400m-patch14-384')
    p.add_argument('--text-embeddings', type=Path, help='Optional .pt dictionary mapping exact query/canonical strings to vectors')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if not np.isfinite(args.temperature) or args.temperature <= 0:
        p.error('temperature must be positive and finite')
    if not args.thresholds or any(not np.isfinite(t) or not 0 <= t <= 1 for t in args.thresholds):
        p.error('thresholds must be finite and in [0, 1]')
    if len(set(args.thresholds)) != len(args.thresholds):
        p.error('thresholds must be unique')
    if any(not np.isclose(round(t * 100), t * 100) for t in args.thresholds):
        p.error('thresholds must be multiples of 0.01 for results_XX folder names')
    return args


def load_scene(root, feature_path):
    import torch
    scene = {k: np.load(root / (k + '.npy')) for k in ['coord', 'scale', 'quat', 'opacity']}
    n = len(scene['coord'])
    for key, shape in [('coord', (n, 3)), ('scale', (n, 3)), ('quat', (n, 4))]:
        if scene[key].shape != shape or not np.isfinite(scene[key]).all():
            raise ValueError(f'Invalid {key}: expected finite {shape}')
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


def load_frames(root, scene_name):
    cameras, images = load_camera_data(root / scene_name / 'sparse' / '0')
    by_name = {image.name: image for image in images.values()}
    frames = []
    for path in sorted((root / 'label' / scene_name).glob('*.json')):
        annotation = json.loads(path.read_text())
        info = annotation['info']
        name = info['name']
        image = by_name[name]
        camera = cameras[image.camera_id]
        if camera.model not in ('PINHOLE', 'SIMPLE_PINHOLE'):
            raise ValueError('Use undistorted sparse/0 cameras and images; lens distortion is unsupported')
        with Image.open(root / scene_name / 'images' / name) as rgb:
            width, height = rgb.size
        if (width, height) != (info['width'], info['height']):
            raise ValueError(f'Annotation/image dimensions differ for {name}')
        fx, fy, cx, cy = get_camera_intrinsics(camera)
        camera = camera._replace(model='PINHOLE', width=width, height=height,
            params=np.array([fx * width / camera.width, fy * height / camera.height,
                             cx * width / camera.width, cy * height / camera.height]))
        import cv2
        masks = {}
        for obj in annotation['objects']:
            mask = masks.setdefault(obj['category'], np.zeros((height, width), dtype=np.uint8))
            polygon = np.asarray(obj['segmentation'], dtype=np.int32)
            if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
                raise ValueError(f'Invalid polygon in {path}')
            cv2.fillPoly(mask, [polygon], 1)
        frames.append(dict(name=name, image=image, camera=camera, masks=masks, shape=(height, width)))
    if not frames:
        raise ValueError('No annotations found under dataset-root/label/scene')
    return frames


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


def query_relevance(features, embeddings, temperature=10.0, device='cuda', chunk_size=32768):
    """Minimum canonical-pair probability, with bounded float32 working memory."""
    import torch
    if features.shape[1] != embeddings.shape[1]:
        raise ValueError('Language/text feature dimensions differ')
    embeddings = torch.nn.functional.normalize(embeddings.float(), dim=-1).to(device)
    scores = []
    with torch.inference_mode():
        for chunk in features.split(chunk_size):
            chunk = torch.nn.functional.normalize(chunk.to(device=device, dtype=torch.float32), dim=-1)
            logits = chunk @ embeddings.T
            scores.append(torch.sigmoid(temperature * (logits[:, 0] - logits[:, 1:].max(dim=1).values)).cpu())
    return torch.cat(scores).numpy()


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


def evaluate_query_threshold(threshold, scores, query, frames, scene):
    selected = scores >= threshold
    indices = np.flatnonzero(selected)
    results = []
    for frame in frames:
        with contextlib.redirect_stdout(io.StringIO()):
            mask, _ = render_semantic_mask(**scene, selected_indices=indices,
                image=frame['image'], camera=frame['camera'], image_shape=frame['shape'])
        results.append((frame, mask, iou(mask, frame['masks'][query])))
    return float(np.mean([r[2] for r in results])), len(indices), results


def main():
    args = parse_args()
    import torch
    if torch.device(args.device).type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Run evaluation inside a GPU allocation or use --device cpu.')
    frames = load_frames(args.dataset_root, args.scene)
    queries = sorted({q for frame in frames for q in frame['masks']})
    if args.query:
        if args.query not in queries:
            raise ValueError(f'Unknown query {args.query!r}; available: {queries}')
        queries = [args.query]
    if not queries:
        raise ValueError('Annotations contain no query objects')
    scene, features = load_scene(args.input_root, args.feature_path)
    text = encode_text(queries + CANONICAL, args)
    if features.shape[1] != text.shape[1]:
        raise ValueError(f'Language/text dimensions differ: {features.shape[1]} vs {text.shape[1]}; select the checkpoint teacher text model')
    scored_queries = []
    for qi, query in enumerate(queries):
        print(f'[{qi + 1}/{len(queries)}] Scoring {query!r}', flush=True)
        embeddings = text[[qi] + list(range(len(queries), len(text)))]
        scores = query_relevance(features, embeddings, args.temperature, args.device)
        quantiles = dict(zip(['min', 'p50', 'p90', 'p99', 'max'],
                             map(float, np.quantile(scores, [0, .5, .9, .99, 1]))))
        query_frames = [f for f in frames if query in f['masks']]
        scored_queries.append((qi, query, scores, quantiles, query_frames))

    for threshold in args.thresholds:
        output_dir = args.output_dir / 'lerf_new_runs' / f'results_{round(threshold * 100):02d}' / args.scene
        output_dir.mkdir(parents=True, exist_ok=True)
        rows, summaries = [], []
        coloured_entries = []
        for qi, query, scores, quantiles, query_frames in scored_queries:
            mean, count, results = evaluate_query_threshold(threshold, scores, query, query_frames, scene)
            mean_macc = float(np.mean([
                binary_macc(mask, frame['masks'][query])
                for frame, mask, _ in results]))
            summary = dict(query=query, mean_iou=mean, mAcc=mean_macc, threshold=threshold,
                           selected_gaussians=count, frames=len(results), score_quantiles=quantiles,
                           threshold_source='fixed')
            summary['artifacts'] = save_query_artifacts(
                output_dir, query, queries, scores, summary, results, scene, args)
            annotated_dir = output_dir / query_folder(query, queries) / 'annotated_frame_masks'
            annotated_dir.mkdir(parents=True, exist_ok=True)
            evaluated_masks = {frame['name']: mask for frame, mask, _ in results}
            indices = np.flatnonzero(scores >= threshold)
            for frame_index, frame in enumerate(frames):
                mask = evaluated_masks.get(frame['name'])
                if mask is None:
                    if len(indices):
                        with contextlib.redirect_stdout(io.StringIO()):
                            mask, _ = render_semantic_mask(
                                **scene, selected_indices=indices,
                                image=frame['image'], camera=frame['camera'],
                                image_shape=frame['shape'])
                    else:
                        mask = np.zeros(frame['shape'], dtype=np.uint8)
                filename = f'{frame_index:04d}_{safe_name(Path(frame["name"]).stem)}.png'
                Image.fromarray(mask.astype(np.uint8) * 255).save(annotated_dir / filename)
                coloured_entries.append((query, frame['name'], annotated_dir / filename))
            summary['artifacts']['annotated_frame_masks'] = (
                annotated_dir.relative_to(output_dir).as_posix())
            summary['annotated_frame_mask_count'] = len(frames)

            if results:
                best_index = max(range(len(results)), key=lambda index: results[index][2])
                best_frame, _, best_iou = results[best_index]
                best_dir = output_dir / 'best_visualization'
                best_dir.mkdir(exist_ok=True)
                best_name = (f'{query_folder(query, queries)}_'
                             f'{safe_name(Path(best_frame["name"]).stem)}_'
                             f'iou_{best_iou:.4f}.png')
                best_path = best_dir / best_name
                shutil.copyfile(
                    output_dir / summary['artifacts']['visualizations'][best_index],
                    best_path)
                summary['artifacts']['best_visualization'] = (
                    best_path.relative_to(output_dir).as_posix())
            summaries.append(summary)
            for fi, (frame, mask, score) in enumerate(results):
                filename = f'q{qi:03d}_f{fi:03d}.png'
                Image.fromarray(mask * 255).save(output_dir / filename)
                rows.append(dict(query=query, image=frame['name'], iou=score,
                                 mAcc=binary_macc(mask, frame['masks'][query]),
                                 threshold=threshold,
                                 mask=filename, predicted_pixels=int(mask.sum()),
                                 gt_pixels=int(frame['masks'][query].sum()),
                                 intersection_pixels=int(np.logical_and(mask, frame['masks'][query]).sum())))
            print(f'{query}: mean IoU={mean:.4f}, threshold={threshold:.2f}, selected={count:,}/{len(features):,}', flush=True)
        report = dict(scene=args.scene, protocol='fixed_threshold', threshold=threshold,
                      voting='none', renderer='scenesplat_selected_gaussian_footprints',
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
        save_coloured_mask_results(
            output_dir, args.dataset_root / args.scene / 'images', coloured_entries)
        print(f'Threshold {threshold:.2f}: query mIoU={report["query_macro_miou"]:.4f}; saved {output_dir}', flush=True)


if __name__ == '__main__':
    main()
