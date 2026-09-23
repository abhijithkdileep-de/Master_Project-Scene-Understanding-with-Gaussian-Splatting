"""Chorus-style comparison artifacts for Chorus 3D OVS evaluation.

All manifest paths are relative to the scene output directory. Gaussian indices
refer to evaluation geometry rows, after applying any feature index sidecar.
"""
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw
from tools.overall_metrics import write_overall_metrics


def safe_name(value):
    name = re.sub(r'[^A-Za-z0-9_.-]+', '_', value).strip('._') or 'query'
    return name


def query_folder(query, all_queries):
    name = safe_name(query)
    if sum(safe_name(q) == name for q in all_queries) > 1:
        name += '_' + hashlib.sha256(query.encode()).hexdigest()[:10]
    return name


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def save_panel(rgb_path, gt, prediction, target, title):
    with Image.open(rgb_path) as source:
        rgb = np.array(source.convert('RGB'))
    overlay = rgb.copy()
    for mask, color in [(gt, [0,255,0]), (prediction, [255,0,0])]:
        active = mask.astype(bool)
        overlay[active] = (.6 * overlay[active] + .4 * np.array(color)).astype(np.uint8)
    h, w = gt.shape
    divider = 12
    panel_x = [0, w, 2*w + divider, 3*w + divider]
    canvas = Image.new('RGB', (4*w + divider, h+64), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), title.encode('ascii', 'replace').decode(), fill='black')
    panels = [rgb, np.repeat((gt*255)[:,:,None],3,axis=2),
              np.repeat((prediction*255)[:,:,None],3,axis=2), overlay]
    for i, (label, panel) in enumerate(zip(['RGB Image','Ground Truth','Prediction','Overlay: GT green, prediction red'],panels)):
        canvas.paste(Image.fromarray(panel.astype(np.uint8)),(panel_x[i],64))
        draw.text((panel_x[i]+8,32),label,fill='black')
    canvas.save(target)


def save_histogram(scores, path):
    counts, edges = np.histogram(scores, bins=100, range=(-1, 1))
    canvas = Image.new('RGB', (1000, 460), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, count in enumerate(counts):
        height = int(350 * count / max(1, counts.max()))
        draw.rectangle((50 + 9*i, 400-height, 57 + 9*i, 400), fill=(65, 110, 165))
    draw.text((50, 12), 'Cosine similarity to scene class', fill='black')
    draw.text((50, 420), '-1                 Cosine similarity (linear count axis)                 1', fill='black')
    canvas.save(path)
    return counts, edges


def save_query_artifacts(output_dir, query, all_queries, scores, selection, summary, results, scene, args):
    output_dir = Path(output_dir)
    slug = query_folder(query, all_queries)
    folder = output_dir / slug
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    def artifact(key, filename):
        path = folder / filename
        paths[key] = path.relative_to(output_dir).as_posix()
        return path
    selection = np.asarray(selection, dtype=np.uint8)
    if selection.shape != scores.shape or np.any(selection > 1):
        raise ValueError('Semantic selection must be one binary value per Gaussian')
    indices = np.flatnonzero(selection)
    indices = indices[np.argsort(-scores[indices], kind='stable')]
    np.save(artifact('cosine_similarity',f'{slug}_cosine_similarity.npy'),scores)
    np.save(artifact('mask',f'{slug}_mask.npy'),selection)
    colors = np.full((len(scores),3),180,dtype=np.uint8)
    colors[selection.astype(bool)] = [255,0,0]
    np.save(artifact('colors',f'{slug}_colors.npy'),colors)
    write_json(artifact('retrieval',f'{slug}_retrieval.json'), dict(
        method='Chorus', scene=args.scene, query=query, selection_method='scene_class_cosine_argmax',
        neighbor_voting=False, vote_k=None,
        num_selected=len(indices),
        selected_fraction=len(indices)/len(scores), gaussian_indices=indices.tolist(),
        cosine_similarities=scores[indices].tolist(), index_space='evaluation_geometry_rows'))
    counts, edges = save_histogram(scores,artifact('histogram',f'{slug}_cosine_similarity_histogram.png'))
    np.savez(artifact('histogram_data',f'{slug}_cosine_similarity_histogram.npz'),counts=counts,edges=edges)
    # A binary XYZ/RGB point cloud for inspecting cosine similarity.
    import cv2
    lo, hi = float(scores.min()), float(scores.max())
    normalized = (scores-lo)/(hi-lo) if hi>lo else np.zeros_like(scores)
    color = cv2.applyColorMap((normalized*255).astype(np.uint8).reshape(-1,1),cv2.COLORMAP_VIRIDIS)[:,0,::-1]
    vertices = np.empty(len(scores),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    for i, key in enumerate(['x','y','z']): vertices[key] = scene['coord'][:,i]
    for i, key in enumerate(['red','green','blue']): vertices[key] = color[:,i]
    with artifact('cosine_similarity_ply',f'{slug}_cosine_similarity.ply').open('wb') as stream:
        stream.write(('ply\nformat binary_little_endian 1.0\nelement vertex '+str(len(scores))+
            '\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode())
        vertices.tofile(stream)
    image_csv = artifact('image_results',f'{slug}_image_results.csv')
    visualizations = []
    with image_csv.open('w',newline='') as stream:
        writer=csv.writer(stream); writer.writerow(['Image','IoU'])
        for i,(frame,mask,score) in enumerate(results):
            writer.writerow([frame['name'],round(score,6)])
            # Index prevents collisions between image basenames from subfolders.
            image_stem=f'{slug}_{i:03d}_{safe_name(Path(frame["name"]).stem)}'
            for subfolder, image in [('prediction_masks',mask),('gt_masks',frame['masks'][query])]:
                dest=folder/subfolder; dest.mkdir(exist_ok=True)
                Image.fromarray(image*255).save(dest/(image_stem+'.png'))
            dest=folder/'mask_visualizations'; dest.mkdir(exist_ok=True)
            panel=dest/(image_stem+'.png')
            save_panel(args.dataset_root/args.scene/'images'/frame['name'],frame['masks'][query],mask,panel,
                       f'{query} | {frame["name"]} | IoU={score:.6f}')
            visualizations.append(panel.relative_to(output_dir).as_posix())
    paths['visualizations']=visualizations
    return paths


def save_comparison_reports(output_dir, report):
    output_dir=Path(output_dir)
    rows=report['queries']
    values=np.array([r['mean_iou'] for r in rows],dtype=np.float64)
    metrics=dict(method='Chorus',scene=report['scene'],protocol=report['protocol'],neighbor_voting=False,
        queries=len(rows),mean_iou=float(values.mean()),median_iou=float(np.median(values)),
        std_iou=float(values.std()),min_iou=float(values.min()),max_iou=float(values.max()),
        per_query={r['query']:r['mean_iou'] for r in rows})
    metrics['mAcc']=float(np.mean([r['mAcc'] for r in rows]))
    metrics['mAcc_definition']='Mean of foreground and background pixel accuracy per annotated query frame; absent GT classes are excluded; macro average over frames, then queries.'
    metrics['per_query_mAcc']={r['query']:r['mAcc'] for r in rows}
    metrics['per_query_metrics']={r['query']:dict(mean_iou=r['mean_iou'],mAcc=r['mAcc'],frames=r['frames']) for r in rows}
    metrics['frame_query_mIoU']=report['frame_query_miou']
    metrics['frame_query_mAcc']=report['frame_query_macc']
    metrics['precision@0.25']=float((values>=.25).mean())
    metrics['precision@0.50']=float((values>=.5).mean())
    write_json(output_dir/'evaluation_metrics.json',metrics)
    write_overall_metrics(output_dir.parent)
    fields=['Method','Scene','Query','Mean_IoU','mAcc','Selected_Gaussians']
    with (output_dir/'per_query_results.csv').open('w',newline='') as stream:
        writer=csv.writer(stream); writer.writerow(fields)
        for row in rows:
            writer.writerow(['Chorus',report['scene'],row['query'],round(row['mean_iou'],6),round(row['mAcc'],6),
                row['selected_gaussians']])
    manifest=dict(method='Chorus',scene=report['scene'],protocol=report['protocol'],neighbor_voting=False,
        num_queries=len(rows),queries=[r['query'] for r in rows],
        retrieval_files=[r['artifacts']['retrieval'] for r in rows],
        mask_files=[r['artifacts']['mask'] for r in rows],color_files=[r['artifacts']['colors'] for r in rows],
        artifacts={r['query']:r['artifacts'] for r in rows},arguments=report['arguments'],
        notes={'precision@IoU':'Fraction of query-mean IoUs meeting the cutoff, matching Chorus naming.',
               'paths':'Relative to this scene output directory.'})
    write_json(output_dir/'evaluation_summary.json',manifest)

