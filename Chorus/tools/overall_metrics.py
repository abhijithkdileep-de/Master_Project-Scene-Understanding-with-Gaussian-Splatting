"""Collect complete per-scene evaluation metrics without recomputing scores."""
import argparse
import json
import os
from pathlib import Path
import tempfile


def write_overall_metrics(output_dir, metric_files=None):
    output_dir = Path(output_dir)
    files = (sorted(output_dir.glob('*/evaluation_metrics.json'))
             if metric_files is None else sorted(map(Path, metric_files)))
    if not files:
        raise ValueError(f'No scene evaluation_metrics.json files found for {output_dir}')

    scenes = {}
    thresholds = set()
    for path in files:
        metrics = json.loads(path.read_text(encoding='utf-8'))
        scene = metrics['scene']
        if scene in scenes:
            raise ValueError(f'Duplicate scene {scene!r}: {path}')
        scenes[scene] = metrics
        if 'threshold' in metrics:
            thresholds.add(metrics['threshold'])
    if len(thresholds) > 1:
        raise ValueError(f'Mixed thresholds in {output_dir}: {sorted(thresholds)}')

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / 'overall_evaluation_metrics.json'
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=output_dir,
                                     prefix='.overall_metrics_', suffix='.tmp',
                                     delete=False) as stream:
        temporary = Path(stream.name)
        json.dump({'scenes': scenes}, stream, indent=2, allow_nan=False)
    os.replace(temporary, destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--scene-first', action='store_true',
                        help='Read <root>/<scene>/threshold_XX/evaluation_metrics.json')
    parser.add_argument('--thresholds', type=int, nargs='+',
                        help='Threshold percentages, required with --scene-first')
    args = parser.parse_args()
    if args.scene_first:
        if not args.thresholds:
            parser.error('--thresholds is required with --scene-first')
        for threshold in args.thresholds:
            tag = f'{threshold:02d}'
            files = args.root.glob(f'*/threshold_{tag}/evaluation_metrics.json')
            print(write_overall_metrics(args.root / f'threshold_{tag}', files))
    else:
        if args.thresholds:
            parser.error('--thresholds requires --scene-first')
        print(write_overall_metrics(args.root))


if __name__ == '__main__':
    main()
