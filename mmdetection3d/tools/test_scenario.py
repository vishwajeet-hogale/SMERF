"""
Scenario-based evaluation for SMERF TopoNet on OpenLane-V2.
Evaluates model performance across different scenario categories (curvature, lighting, occlusion, etc.)
with per-scenario metrics, visualizations, and aggregations.
Supports memory-efficient streaming and caching.
"""
import argparse
import os
import os.path as osp
import warnings
from collections import defaultdict, OrderedDict
from datetime import datetime

import cv2
import matplotlib
import mmcv
import numpy as np
import pandas as pd
import torch
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from mmdet3d.apis import single_gpu_test

# ===================================================
# Configuration & Constants
# ===================================================

METRIC_LABELS = {
    'OpenLane-V2 Score': 'OLV2 Score',
    'DET_l': 'DET$_l$',
    'DET_t': 'DET$_t$',
    'TOP_ll': 'TOP$_{ll}$',
    'TOP_lt': 'TOP$_{lt}$',
}
METRIC_COLS = list(METRIC_LABELS.keys())
SCENARIO_TYPES = {
    'curvature': 'Road Curvature',
    'lighting': 'Lighting Condition',
    'occlusion': 'Occlusion Level',
    'topology_complexity': 'Topology Complexity',
}
PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
METRIC_COLS_EVAL = ['OpenLane-V2 Score', 'DET_l', 'DET_t', 'TOP_ll', 'TOP_lt']


# ===================================================
# Memory & Cache Utilities
# ===================================================

def _format_bytes(num_bytes):
    """Format bytes as human-readable string."""
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0:
            return f'{value:.1f} {unit}'
        value /= 1024.0
    return f'{value:.1f} PB'


def _get_available_memory_bytes():
    """Get available system memory in bytes."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.available
    except ImportError:
        pass

    # Fallback: try os.sysconf
    try:
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_AVPHYS_PAGES')
    except (AttributeError, ValueError):
        pass

    return None


def _get_sample_scenario_labels(info):
    """Extract scenario labels from sample metadata."""
    out = {}
    meta = info.get('scenario_meta', {})

    if 'curvature' in meta:
        value = meta['curvature'].get('value_m_inv')
        thresholds = meta['curvature'].get('thresholds_m_inv', {})
        if value is not None:
            if value <= thresholds.get('straight', 0.003):
                out['curvature'] = 'straight'
            elif value <= thresholds.get('low', 0.008):
                out['curvature'] = 'low curvature'
            elif value <= thresholds.get('medium', 0.02):
                out['curvature'] = 'medium curvature'
            else:
                out['curvature'] = 'high curvature'

    if 'topology_complexity' in meta:
        value = meta['topology_complexity'].get('numeric_level')
        if value is not None:
            if value == 1:
                out['topology_complexity'] = 'simple'
            elif value == 2:
                out['topology_complexity'] = 'moderate'
            else:
                out['topology_complexity'] = 'complex'

    if 'lighting' in meta:
        label = meta['lighting'].get('label')
        if label:
            out['lighting'] = label

    if 'occlusion' in meta:
        label = meta['occlusion'].get('label')
        if label:
            out['occlusion'] = label

    return out


def parse_args():
    parser = argparse.ArgumentParser(description='SMERF scenario-based evaluation')

    parser.add_argument('config', help='config file')
    parser.add_argument('checkpoint', help='checkpoint file')

    parser.add_argument('--out', action='store_true', help='save outputs to pickle')
    parser.add_argument('--out-dir', default='work_dirs/results',
                        help='directory to save outputs')
    parser.add_argument('--eval', nargs='+', help='evaluation metrics')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=-1,
                        help='limit samples (use -1 for all)')
    parser.add_argument('--workers-per-gpu', type=int, default=0,
                        help='dataloader workers per GPU')
    parser.add_argument('--stream-out', action='store_true',
                        help='stream per-sample predictions to disk')
    parser.add_argument('--stream-dir', default='',
                        help='directory for streamed predictions')
    parser.add_argument('--clear-cache-interval', type=int, default=10,
                        help='clear cache every N batches (<=0 disables)')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU id to use')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--show-scenario-plots', action='store_true',
                        help='generate scenario comparison plots')
    parser.add_argument('--memory-monitor', action='store_true',
                        help='monitor GPU/system memory usage during inference')

    args = parser.parse_args()
    return args


def _bar_chart_per_scenario(df, scenario_type, global_score, run_dir):
    """Create grouped bar chart for metrics per scenario category."""
    if df.empty:
        return

    categories = df.index.tolist()
    n_cats = len(categories)
    n_metrics = len(METRIC_COLS)
    x = np.arange(n_cats)
    width = 0.15

    fig, ax = plt.subplots(figsize=(max(7, n_cats * 2.2), 5))
    for j, metric in enumerate(METRIC_COLS):
        offset = width * (j - (n_metrics - 1) / 2)
        values = df[metric].values
        ax.bar(x + offset, values, width, label=METRIC_LABELS[metric])

    ax.axhline(y=global_score, color='grey', linestyle='--', linewidth=1,
               label=f'Global OLV2 ({global_score:.3f})')
    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(f'Scenario: {SCENARIO_TYPES.get(scenario_type, scenario_type)}',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, min(1.0, df[METRIC_COLS].max().max() + 0.12))
    ax.legend(fontsize=8, ncol=3, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(osp.join(run_dir, f'bar_{scenario_type}.png'), dpi=200)
    fig.savefig(osp.join(run_dir, f'bar_{scenario_type}.pdf'))
    plt.close(fig)


def _score_drop_chart(scenario_tables, global_score, run_dir):
    """Horizontal bar chart showing performance delta from global."""
    rows = []
    for stype, df in scenario_tables.items():
        for category, row in df.iterrows():
            delta = float(row['OpenLane-V2 Score']) - global_score
            rows.append({
                'category': f"{category} ({stype})",
                'delta': delta,
                'samples': int(row.get('samples', 0))
            })

    if not rows:
        return

    delta_df = pd.DataFrame(rows).sort_values('delta')

    fig, ax = plt.subplots(figsize=(8, max(4, len(rows) * 0.55)))
    colors = ['#C44E52' if d < 0 else '#55A868' for d in delta_df['delta']]
    labels = [f"{r['category']}" for _, r in delta_df.iterrows()]
    bars = ax.barh(range(len(delta_df)), delta_df['delta'], color=colors,
                   edgecolor='white', height=0.6)
    ax.set_yticks(range(len(delta_df)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(x=0, color='grey', linewidth=0.8)
    ax.set_xlabel('$\\Delta$ OLV2 Score (vs. global)', fontsize=11)
    ax.set_title('Performance Gap by Scenario Category', fontsize=13, fontweight='bold')
    
    for bar, v in zip(bars, delta_df['delta']):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f'{v:.3f}', ha='left' if v > 0 else 'right',
                va='center', fontsize=8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(osp.join(run_dir, 'score_delta.png'), dpi=200)
    fig.savefig(osp.join(run_dir, 'score_delta.pdf'))
    plt.close(fig)


def _heatmap(scenario_tables, run_dir):
    """Heatmap showing all metrics across all scenarios."""
    rows = []
    labels = []
    for stype, df in scenario_tables.items():
        for category, row in df.iterrows():
            labels.append(f"{category} ({stype[:3].upper()})")
            rows.append([float(row[m]) for m in METRIC_COLS])

    if not rows:
        return

    mat = np.array(rows)

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.5)))
    im = ax.imshow(mat, aspect='auto', cmap='RdYlGn',
                   vmin=0, vmax=max(0.6, mat.max() + 0.05))
    ax.set_xticks(range(len(METRIC_COLS)))
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRIC_COLS], fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            text = ax.text(j, i, f'{mat[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)

    ax.set_title('Metric Heatmap Across All Scenarios', fontsize=13, fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.8, label='Score')
    fig.tight_layout()
    fig.savefig(osp.join(run_dir, 'heatmap.png'), dpi=200)
    fig.savefig(osp.join(run_dir, 'heatmap.pdf'))
    plt.close(fig)


def generate_visualizations(scenario_tables, global_score, run_dir):
    """Generate all visualization plots."""
    print("\n" + "="*50)
    print("GENERATING VISUALIZATIONS")
    print("="*50)
    
    for stype, df in scenario_tables.items():
        _bar_chart_per_scenario(df, stype, global_score, run_dir)

    _score_drop_chart(scenario_tables, global_score, run_dir)
    _heatmap(scenario_tables, run_dir)

    print(f"\nVisualizations saved to: {run_dir}")
    print("  bar_<scenario>.png/pdf        : grouped bar charts per scenario")
    print("  score_delta.png/pdf           : performance gap vs. global")
    print("  heatmap.png/pdf               : metric heatmap across all categories")


def _sanitize_eval_kwargs(eval_kwargs):
    """Remove EvalHook-only keys that dataset.evaluate doesn't accept."""
    cleaned = dict(eval_kwargs) if eval_kwargs is not None else {}
    for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
        cleaned.pop(key, None)
    return cleaned


def run_inference(model, data_loader, stream_out=False, stream_dir=None,
                  clear_cache_interval=10, memory_monitor=False):
    """Run inference loop with optional streaming and memory monitoring."""
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    outputs = []
    streamed_files = []
    stream_idx = 0

    if stream_out and stream_dir:
        mmcv.mkdir_or_exist(stream_dir)

    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(data_loader):
            with torch.cuda.amp.autocast(enabled=False):
                result = model(return_loss=False, rescale=True, **data)
            outputs.append(result)

            if stream_out and stream_dir:
                pkl_path = osp.join(stream_dir, f'sample_{stream_idx:07d}.pkl')
                mmcv.dump(result, pkl_path)
                streamed_files.append(pkl_path)
                stream_idx += 1

            if memory_monitor and (idx + 1) % max(1, clear_cache_interval) == 0:
                gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3)
                gpu_max = torch.cuda.max_memory_allocated() / (1024 ** 3)
                print(f'[Batch {idx+1}] GPU Memory: {gpu_mem:.2f}GB / Peak: {gpu_max:.2f}GB')

            if clear_cache_interval > 0 and (idx + 1) % clear_cache_interval == 0:
                torch.cuda.empty_cache()

            prog_bar.update()

    return outputs, streamed_files


def evaluate_by_scenario(dataset, outputs, eval_kwargs, out_dir):
    """Evaluate metrics per scenario category."""
    print("\n" + "="*50)
    print("EVALUATING BY SCENARIO")
    print("="*50)

    # Global evaluation
    print("\n--- GLOBAL METRICS ---")
    global_metrics = dataset.evaluate(outputs, **eval_kwargs)
    global_score = float(global_metrics.get('OpenLane-V2 Score', 0))
    print(f"Global OLV2 Score: {global_score:.4f}")

    # Scenario-based evaluation
    scenario_indices = defaultdict(lambda: defaultdict(list))
    for idx, info in enumerate(dataset.data_infos):
        labels = _get_sample_scenario_labels(info)
        for scenario_type, category in labels.items():
            scenario_indices[scenario_type][category].append(idx)

    scenario_tables = {}
    all_rows = []

    for scenario_type in sorted(scenario_indices.keys()):
        print(f"\n--- {SCENARIO_TYPES.get(scenario_type, scenario_type).upper()} ---")
        categories = scenario_indices[scenario_type]
        category_metrics = {}

        for category in sorted(categories.keys()):
            indices = categories[category]
            print(f"\n{category} (n={len(indices)})")

            # Evaluate subset
            original_infos = dataset.data_infos
            try:
                dataset.data_infos = [original_infos[i] for i in indices]
                subset_outputs = [outputs[i] for i in indices]
                subset_metrics = dataset.evaluate(subset_outputs, **eval_kwargs)
                
                category_metrics[category] = subset_metrics
                for metric_name, value in subset_metrics.items():
                    print(f"  {metric_name}: {value:.4f}")
                    all_rows.append({
                        'scenario_type': scenario_type,
                        'category': category,
                        'samples': len(indices),
                        'metric': metric_name,
                        'value': float(value)
                    })
            finally:
                dataset.data_infos = original_infos

        # Create dataframe for this scenario type
        if category_metrics:
            rows = []
            for category, metrics in category_metrics.items():
                row = {'category': category, 'samples': len(categories[category])}
                for metric in METRIC_COLS_EVAL:
                    row[metric] = float(metrics.get(metric, 0))
                rows.append(row)
            df = pd.DataFrame(rows).set_index('category')
            scenario_tables[scenario_type] = df
            print(f"\n{scenario_type.upper()} Results:")
            print(df)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = osp.join(out_dir, f'eval_{timestamp}')
    mmcv.mkdir_or_exist(run_dir)

    # Save global metrics
    global_row = {'scenario_type': 'global', 'category': 'all', 'samples': len(outputs)}
    for metric in METRIC_COLS_EVAL:
        global_row[metric] = float(global_metrics.get(metric, 0))

    global_df = pd.DataFrame([global_row]).set_index('scenario_type')
    global_df.to_csv(osp.join(run_dir, 'global_metrics.csv'))

    # Save scenario results
    for scenario_type, df in scenario_tables.items():
        df.to_csv(osp.join(run_dir, f'scenario_{scenario_type}.csv'))

    # Save combined results
    if all_rows:
        combined_df = pd.DataFrame(all_rows)
        combined_df.to_csv(osp.join(run_dir, 'all_metrics.csv'), index=False)

    # Generate visualizations
    generate_visualizations(scenario_tables, global_score, run_dir)

    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)
    print(f"Results saved to: {run_dir}")
    print(f"  global_metrics.csv")
    print(f"  scenario_*.csv")
    print(f"  all_metrics.csv")
    print(f"  *.png, *.pdf (visualizations)")

    return run_dir


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Set GPU
    cfg.gpu_ids = [args.gpu_id]

    # Set random seed
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=False)

    # Override split if needed
    if args.max_samples > 0:
        # Note: this is a hint, actual limiting happens in dataset
        pass

    # Build dataset
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)

    # Limit samples if requested
    if args.max_samples > 0:
        dataset.data_infos = dataset.data_infos[:args.max_samples]

    # Build dataloader
    dataloader_cfg = {
        'samples_per_gpu': 1,
        'workers_per_gpu': args.workers_per_gpu,
        'dist': False,
        'shuffle': False
    }
    data_loader = build_dataloader(dataset, **dataloader_cfg)

    # Build model
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    model = MMDataParallel(model, device_ids=cfg.gpu_ids)

    # Run inference
    print("\n" + "="*50)
    print("RUNNING INFERENCE")
    print("="*50)
    outputs, streamed_files = run_inference(
        model, data_loader,
        stream_out=args.stream_out,
        stream_dir=args.stream_dir if args.stream_dir else None,
        clear_cache_interval=args.clear_cache_interval,
        memory_monitor=args.memory_monitor
    )

    print(f"\nInference complete. Generated {len(outputs)} outputs.")

    if args.out:
        out_path = osp.join(args.out_dir, 'inference_outputs.pkl')
        mmcv.mkdir_or_exist(args.out_dir)
        mmcv.dump(outputs, out_path)
        print(f"Outputs saved to: {out_path}")

    # Evaluate
    eval_kwargs = _sanitize_eval_kwargs(cfg.get('evaluation', {}).copy())
    if args.eval is not None:
        eval_kwargs['metrics'] = args.eval

    mmcv.mkdir_or_exist(args.out_dir)
    evaluate_by_scenario(dataset, outputs, eval_kwargs, args.out_dir)


if __name__ == '__main__':
    main()
