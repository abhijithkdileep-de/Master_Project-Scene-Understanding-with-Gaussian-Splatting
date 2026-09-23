"""Create LERF prediction-only RGB cutouts from already saved binary masks."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw


def safe_name(value):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', value).strip('._') or 'item'


def save_coloured_mask_results(scene_output_dir, image_root, entries):
    """Save one full-resolution cutout per query/frame and an RGB/query overview.

    Entries are (query, image_name, saved_binary_mask_path). No evaluation arrays
    or scores are modified here.
    """
    scene_output_dir, image_root = Path(scene_output_dir), Path(image_root)
    output = scene_output_dir / 'Coloured Mask results'
    output.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    entries = list(entries)
    queries = list(dict.fromkeys(query for query, _, _ in entries))
    slugs = {query: safe_name(query) for query in queries}
    for query in queries:
        if sum(slugs[other] == slugs[query] for other in queries) > 1:
            slugs[query] += '_' + hashlib.sha256(query.encode()).hexdigest()[:10]
    frame_numbers = {name: index for index, name in enumerate(
        dict.fromkeys(name for _, name, _ in entries))}
    records = []
    for query, image_name, mask_path in entries:
        with Image.open(image_root / image_name) as source:
            rgb = np.asarray(source.convert('RGB'))
        with Image.open(mask_path) as source:
            mask = np.asarray(source.convert('L')) > 0
        if mask.shape != rgb.shape[:2]:
            raise ValueError(f'{image_name}: prediction mask and RGB dimensions differ')
        coloured = np.full_like(rgb, 255)
        coloured[mask] = rgb[mask]
        basename = f'{frame_numbers[image_name]:04d}_{safe_name(Path(image_name).stem)}.png'
        target = output / slugs[query] / basename
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(coloured).save(target)
        relative = target.relative_to(scene_output_dir).as_posix()
        records.append(dict(query=query, image=image_name, prediction=relative))
        grouped[image_name].append((query, target))

    overview_dir = output / 'overview'
    overview_dir.mkdir(exist_ok=True)
    overviews = []
    for image_name, predictions in grouped.items():
        with Image.open(image_root / image_name) as source:
            rgb = source.convert('RGB')
        scale = min(1., 480 / max(rgb.size))
        size = (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale)))
        width, gap, margin, header = size[0], 10, 12, 36
        canvas = Image.new('RGB', (2 * margin + (len(predictions) + 1) * width +
                                   len(predictions) * gap, header + size[1] + margin), 'white')
        draw = ImageDraw.Draw(canvas)
        panels = [('RGB', rgb)]
        for query, path in predictions:
            with Image.open(path) as source:
                panels.append((query, source.convert('RGB')))
        for index, (title, panel) in enumerate(panels):
            x = margin + index * (width + gap)
            draw.text((x, 8), title, fill='black')
            canvas.paste(panel.resize(size, Image.Resampling.NEAREST), (x, header))
        target = overview_dir / f'{frame_numbers[image_name]:04d}_{safe_name(Path(image_name).stem)}.png'
        canvas.save(target)
        overviews.append(dict(image=image_name, overview=target.relative_to(scene_output_dir).as_posix()))
    manifest = dict(background='white', mask_source='saved_prediction_masks',
                    queries=queries, predictions=records, overviews=overviews)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return (output / 'manifest.json').relative_to(scene_output_dir).as_posix()

