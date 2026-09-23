"""Comparable, multi-class 3DOVS visualizations on the original image grid."""
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw


# Colors follow the six swatches in the supplied bed and sofa comparison figures.
REFERENCE_PALETTE = (
    (239, 22, 72),   # pink
    (41, 179, 74),   # green
    (255, 226, 38),  # yellow
    (59, 96, 222),   # blue
    (249, 87, 58),   # orange-red
    (142, 18, 190),  # purple
    (0, 187, 190),   # cyan, for scenes with a seventh class
)
REFERENCE_LABEL_COLORS = {
    'banana': REFERENCE_PALETTE[0],
    'black leather shoe': REFERENCE_PALETTE[1],
    'camera': REFERENCE_PALETTE[2],
    'hand': REFERENCE_PALETTE[3],
    'red bag': REFERENCE_PALETTE[4],
    'white sheet': REFERENCE_PALETTE[5],
    'gundam': REFERENCE_PALETTE[0],
    'pikachu': REFERENCE_PALETTE[1],
    'xbox wireless controller': REFERENCE_PALETTE[2],
    'a red nintendo switch joy-con controller': REFERENCE_PALETTE[3],
    'a stack of uno cards': REFERENCE_PALETTE[4],
    'grey sofa': REFERENCE_PALETTE[5],
}


def class_colors(classes):
    """Match reference object colors regardless of the evaluator's class order."""
    if len(classes) > len(REFERENCE_PALETTE):
        raise ValueError('Extend the qualitative palette for scenes with more classes')
    colors = [(0, 0, 0)]
    used = {REFERENCE_LABEL_COLORS[q.strip().lower()] for q in classes
            if q.strip().lower() in REFERENCE_LABEL_COLORS}
    available = iter(color for color in REFERENCE_PALETTE if color not in used)
    for query in classes:
        colors.append(REFERENCE_LABEL_COLORS.get(query.strip().lower()) or next(available))
    return np.asarray(colors, dtype=np.uint8)


def label_image(masks, classes, shape):
    """Compose binary annotation or prediction masks into one class image."""
    labels = np.zeros(shape, dtype=np.uint16)
    for class_id, query in enumerate(classes, start=1):
        mask = masks.get(query)
        if mask is not None:
            if mask.shape != shape:
                raise ValueError(f'{query}: mask shape {mask.shape} differs from {shape}')
            labels[np.asarray(mask, dtype=bool)] = class_id
    return labels


def color_overlay(rgb, labels, palette):
    overlay = rgb.copy()
    active = labels != 0
    overlay[active] = (.35 * rgb[active] + .65 * palette[labels[active]]).astype(np.uint8)
    return overlay


def save_semantic_visualizations(output_dir, frames, results_by_query, classes, args, method):
    """Write class maps and RGB/GT/prediction/error panels for every frame."""
    output_dir = Path(output_dir)
    folder = output_dir / 'Coloured Mask results'
    folder.mkdir(parents=True, exist_ok=True)
    palette = class_colors(classes)
    by_frame = {query: {id(frame): mask for frame, mask, _ in results_by_query[query]}
                for query in classes}
    records = []
    for frame_index, frame in enumerate(frames):
        with Image.open(args.dataset_root / args.scene / 'images' / frame['name']) as image:
            rgb = np.asarray(image.convert('RGB'))
        shape = rgb.shape[:2]
        if shape != frame['shape']:
            raise ValueError(f'{frame["name"]}: image and annotation dimensions differ')
        gt = label_image(frame['masks'], classes, shape)
        prediction = label_image({query: by_frame[query][id(frame)] for query in classes}, classes, shape)
        stem = re.sub(r'[^A-Za-z0-9_.-]+', '_', Path(frame['name']).stem).strip('._') or 'frame'
        basename = f'{frame_index:03d}_{stem}'
        label_path = folder / f'{basename}_labels.png'
        Image.fromarray(prediction).save(label_path)

        # Keep display images manageable while retaining full-resolution label PNGs.
        scale = min(1.0, 640 / max(shape))
        size = (max(1, round(shape[1] * scale)), max(1, round(shape[0] * scale)))
        error = rgb.copy()
        mismatch = gt != prediction
        error[mismatch] = (.3 * rgb[mismatch] + .7 * np.array([255, 210, 0])).astype(np.uint8)
        panels = [rgb, color_overlay(rgb, gt, palette),
                  color_overlay(rgb, prediction, palette), error]
        panels = [Image.fromarray(panel).resize(size, Image.Resampling.NEAREST)
                  for panel in panels]
        margin, gap, header = 12, 10, 48
        legend_rows = (len(classes) + 3) // 4
        canvas = Image.new('RGB', (2 * margin + 4 * size[0] + 3 * gap,
                                   header + size[1] + 18 + legend_rows * 24), 'white')
        draw = ImageDraw.Draw(canvas)
        for index, (title, panel) in enumerate(zip(('RGB', 'Ground truth', method, 'Mismatch'), panels)):
            x = margin + index * (size[0] + gap)
            draw.text((x, 8), title, fill='black')
            canvas.paste(panel, (x, header))
        for index, query in enumerate(classes):
            row, column = divmod(index, 4)
            x = margin + column * (size[0] + gap)
            y = header + size[1] + 14 + row * 24
            draw.rectangle((x, y, x + 14, y + 14), fill=tuple(palette[index + 1]))
            draw.text((x + 20, y), query, fill='black')
        panel_path = folder / f'{basename}_comparison.png'
        canvas.save(panel_path)
        records.append(dict(image=frame['name'], labels=label_path.relative_to(output_dir).as_posix(),
                            comparison=panel_path.relative_to(output_dir).as_posix()))
    manifest = dict(method=method, classes=classes,
                    colors={query: palette[index + 1].tolist() for index, query in enumerate(classes)},
                    background=0, labels_are_one_based=True, frames=records)
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return 'Coloured Mask results/manifest.json'
