"""
Aggregate SMERF predictions by scenario categories with detailed statistics.

Loads streamed predictions, extracts prediction statistics (confidence, counts),
and aggregates them by scenario category to analyze model behavior across conditions.
"""
import argparse
import os
import os.path as osp
from collections import defaultdict
from datetime import datetime
from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mmcv

matplotlib.use('Agg')

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


# ===================================================
# Configuration & Constants
# ===================================================

METRIC_COLS = ['OpenLane-V2 Score', 'DET_l', 'DET_t', 'TOP_ll', 'TOP_lt']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Aggregate SMERF predictions by scenario categories'
    )
    parser.add_argument(
        '--stream-dir',
        default='work_dirs/results/stream_outputs',
        help='Directory containing streamed .pkl files'
    )
    parser.add_argument(
        '--data-root',
        default='data/OpenLane-V2',
        help='Root directory of OpenLane-V2 data'
    )
    parser.add_argument(
        '--split',
        default='val',
        help='Dataset split used for predictions'
    )
    parser.add_argument(
        '--out-dir',
        default='work_dirs/results',
        help='Directory for outputs'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=-1,
        help='Limit number of samples. -1 means all.'
    )
    return parser.parse_args()


def get_curvature_label(meta):
    """Extract curvature category from metadata."""
    if 'curvature' not in meta:
        return None
    
    value = meta['curvature'].get('value_m_inv')
    thresholds = meta['curvature'].get('thresholds_m_inv', {})
    
    if value is None:
        return None
    
    if value <= thresholds.get('straight', 0.003):
        return 'straight'
    elif value <= thresholds.get('low', 0.008):
        return 'low curvature'
    elif value <= thresholds.get('medium', 0.02):
        return 'medium curvature'
    else:
        return 'high curvature'


def get_sample_labels(scenario_meta):
    """Extract all scenario labels for a sample."""
    labels = {}

    curvature = get_curvature_label(scenario_meta)
    if curvature:
        labels['curvature'] = curvature

    lighting = scenario_meta.get('lighting', {}).get('label')
    if lighting:
        labels['lighting'] = lighting

    occlusion = scenario_meta.get('occlusion', {}).get('label')
    if occlusion:
        labels['occlusion'] = occlusion

    topology = scenario_meta.get('topology_complexity', {}).get('numeric_level')
    if topology is not None:
        if topology == 1:
            labels['topology_complexity'] = 'simple'
        elif topology == 2:
            labels['topology_complexity'] = 'moderate'
        else:
            labels['topology_complexity'] = 'complex'

    return labels


def get_prediction_stats(pred):
    """Extract statistics from a prediction dict."""
    stats = {
        'n_lanes': 0,
        'n_traffic_elements': 0,
        'lane_conf_mean': np.nan,
        'te_conf_mean': np.nan,
    }

    lane_scores = []
    te_scores = []

    # Extract lane confidences
    lane_results = pred.get('lane_results')
    if lane_results is not None and len(lane_results) > 1 and lane_results[1] is not None:
        scores = lane_results[1]
        if isinstance(scores, np.ndarray):
            lane_scores = scores.tolist()
        elif isinstance(scores, list):
            lane_scores = scores

    # Extract traffic element confidences
    bbox_results = pred.get('bbox_results')
    if bbox_results is not None and len(bbox_results) > 1 and bbox_results[1] is not None:
        scores = bbox_results[1]
        if isinstance(scores, np.ndarray):
            te_scores = scores.tolist()
        elif isinstance(scores, list):
            te_scores = scores

    stats['n_lanes'] = int(len(lane_scores))
    stats['n_traffic_elements'] = int(len(te_scores))
    stats['lane_conf_mean'] = float(np.mean(lane_scores)) if lane_scores else np.nan
    stats['te_conf_mean'] = float(np.mean(te_scores)) if te_scores else np.nan

    return stats


def load_stream_paths(stream_dir):
    """Load all prediction pkl paths from stream directory."""
    manifest = osp.join(stream_dir, 'manifest.txt')
    paths = []

    if osp.isfile(manifest):
        with open(manifest, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    paths.append(osp.join(stream_dir, line))

    if not paths:
        paths = sorted(glob(osp.join(stream_dir, '*.pkl')))

    if not paths:
        raise ValueError(f'No pkl files found in {stream_dir}')

    return paths


def load_json_paths(data_root, split):
    """Load all JSON file paths for a dataset split."""
    split_dir = osp.join(data_root, split)
    if not osp.isdir(split_dir):
        raise ValueError(f'Split directory not found: {split_dir}')

    paths = []
    segments = sorted(glob(osp.join(split_dir, '*')))
    
    for seg in segments:
        info_dir = osp.join(seg, 'info')
        json_files = sorted(glob(osp.join(info_dir, '*.json')))
        paths.extend(json_files)

    if not paths:
        raise ValueError(f'No JSON files found in {split_dir}')

    return paths


def update_bucket(bucket, stats):
    """Update aggregation bucket with new sample stats."""
    bucket['samples'] += 1
    bucket['sum_lanes'] += stats['n_lanes']
    bucket['sum_traffic_elements'] += stats['n_traffic_elements']

    if not np.isnan(stats['lane_conf_mean']):
        bucket['sum_lane_conf_mean'] += stats['lane_conf_mean']
        bucket['count_lane_conf_mean'] += 1

    if not np.isnan(stats['te_conf_mean']):
        bucket['sum_te_conf_mean'] += stats['te_conf_mean']
        bucket['count_te_conf_mean'] += 1


def finalize_bucket(bucket):
    """Compute final aggregated statistics from bucket."""
    samples = max(bucket['samples'], 1)
    lane_conf_count = max(bucket['count_lane_conf_mean'], 1)
    te_conf_count = max(bucket['count_te_conf_mean'], 1)

    return {
        'samples': bucket['samples'],
        'avg_lanes': bucket['sum_lanes'] / samples,
        'avg_traffic_elements': bucket['sum_traffic_elements'] / samples,
        'avg_lane_conf_mean': bucket['sum_lane_conf_mean'] / lane_conf_count,
        'avg_te_conf_mean': bucket['sum_te_conf_mean'] / te_conf_count,
    }


def _resolve_workers(workers, total_items):
    """Determine optimal number of workers."""
    if total_items <= 1:
        return 1
    if workers and workers > 0:
        return workers
    cpu_count = os.cpu_count() or 1
    return min(max(4, cpu_count), 16, total_items)


def aggregate(stream_paths, json_paths, split, workers=0):
    """Aggregate predictions by scenario category."""
    scenario_buckets = {
        'curvature': defaultdict(lambda: {
            'samples': 0,
            'sum_lanes': 0,
            'sum_traffic_elements': 0,
            'sum_lane_conf_mean': 0.0,
            'count_lane_conf_mean': 0,
            'sum_te_conf_mean': 0.0,
            'count_te_conf_mean': 0,
        }),
        'lighting': defaultdict(lambda: {
            'samples': 0,
            'sum_lanes': 0,
            'sum_traffic_elements': 0,
            'sum_lane_conf_mean': 0.0,
            'count_lane_conf_mean': 0,
            'sum_te_conf_mean': 0.0,
            'count_te_conf_mean': 0,
        }),
        'occlusion': defaultdict(lambda: {
            'samples': 0,
            'sum_lanes': 0,
            'sum_traffic_elements': 0,
            'sum_lane_conf_mean': 0.0,
            'count_lane_conf_mean': 0,
            'sum_te_conf_mean': 0.0,
            'count_te_conf_mean': 0,
        }),
        'topology_complexity': defaultdict(lambda: {
            'samples': 0,
            'sum_lanes': 0,
            'sum_traffic_elements': 0,
            'sum_lane_conf_mean': 0.0,
            'count_lane_conf_mean': 0,
            'sum_te_conf_mean': 0.0,
            'count_te_conf_mean': 0,
        }),
    }

    total = min(len(stream_paths), len(json_paths))
    print(f'\nAggregating {total} samples...')

    # Simple sequential aggregation
    pbar = tqdm(total=total) if tqdm is not None else None
    
    for pkl_path, json_path in zip(stream_paths, json_paths):
        try:
            # Load prediction
            pred = mmcv.load(pkl_path)
            
            # Load JSON metadata
            import json
            with open(json_path, 'r') as f:
                sample_json = json.load(f)
            
            scenario_meta = sample_json.get('scenario_meta', {})
            labels = get_sample_labels(scenario_meta)
            stats = get_prediction_stats(pred)

            # Update scenario buckets
            for scenario_type, category in labels.items():
                if scenario_type in scenario_buckets:
                    update_bucket(scenario_buckets[scenario_type][category], stats)

        except Exception as e:
            print(f'Error processing {pkl_path}: {e}')
            continue

        if pbar is not None:
            pbar.update()

    if pbar is not None:
        pbar.close()

    # Finalize buckets
    final = {}
    for scenario_type, category_map in scenario_buckets.items():
        final[scenario_type] = {
            category: finalize_bucket(bucket)
            for category, bucket in category_map.items()
        }

    return final


def to_dataframe(category_dict):
    """Convert category dict to DataFrame."""
    rows = []
    for category, values in sorted(category_dict.items()):
        row = {'category': category}
        row.update(values)
        rows.append(row)
    
    if not rows:
        return pd.DataFrame()
    
    return pd.DataFrame(rows).set_index('category')


def plot_scenario(df, scenario_type, run_dir):
    """Plot aggregated statistics for a scenario type."""
    if df.empty:
        return

    categories = [f"{cat}\n(n={int(n)})" for cat, n in zip(df.index, df['samples'])]
    x = np.arange(len(categories))

    fig, axes = plt.subplots(2, 1, figsize=(max(8, len(categories) * 1.4), 8), sharex=True)

    # Plot lane and TE counts
    bars_lanes = axes[0].bar(x - 0.2, df['avg_lanes'], width=0.4, 
                             label='Avg Lanes/sample', color='#2a9d8f')
    bars_te = axes[0].bar(x + 0.2, df['avg_traffic_elements'], width=0.4, 
                          label='Avg TEs/sample', color='#457b9d')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'{scenario_type.title()} - Prediction Density By Category')
    axes[0].grid(axis='y', alpha=0.25)
    axes[0].legend(frameon=False)

    for bar in bars_lanes:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    for bar in bars_te:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)

    # Plot confidence scores
    bars_lane_conf = axes[1].bar(x - 0.2, df['avg_lane_conf_mean'], width=0.4, 
                                label='Mean lane confidence', color='#f4a261')
    bars_te_conf = axes[1].bar(x + 0.2, df['avg_te_conf_mean'], width=0.4, 
                              label='Mean TE confidence', color='#e76f51')
    axes[1].set_ylabel('Score')
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title(f'{scenario_type.title()} - Confidence Quality By Category')
    axes[1].grid(axis='y', alpha=0.25)
    axes[1].legend(frameon=False)

    for bar in bars_lane_conf:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    for bar in bars_te_conf:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories, rotation=20, ha='right')
    
    fig.tight_layout()
    fig.savefig(osp.join(run_dir, f'stats_{scenario_type}.png'), dpi=160)
    plt.close(fig)


def main():
    args = parse_args()

    print('\n' + '='*60)
    print('AGGREGATING SMERF PREDICTIONS BY SCENARIO CATEGORIES')
    print('='*60)

    # Load paths
    print(f'\nLoading streamed predictions from: {args.stream_dir}')
    stream_paths = load_stream_paths(args.stream_dir)
    print(f'Found {len(stream_paths)} prediction files')

    print(f'\nLoading JSON metadata from: {args.data_root}/{args.split}')
    json_paths = load_json_paths(args.data_root, args.split)
    print(f'Found {len(json_paths)} JSON files')

    # Limit if needed
    if args.max_samples > 0:
        stream_paths = stream_paths[:args.max_samples]
        json_paths = json_paths[:args.max_samples]
        print(f'Limited to {args.max_samples} samples')

    # Aggregate
    aggregated = aggregate(stream_paths, json_paths, args.split)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = osp.join(args.out_dir, f'category_agg_{timestamp}')
    mmcv.mkdir_or_exist(run_dir)

    print(f'\nSaving results to: {run_dir}')

    # Save CSVs
    for scenario_type, category_dict in aggregated.items():
        if not category_dict:
            continue
        
        df = to_dataframe(category_dict)
        csv_path = osp.join(run_dir, f'category_stats_{scenario_type}.csv')
        df.to_csv(csv_path)
        print(f'  {osp.basename(csv_path)}')
        
        # Generate plot
        plot_scenario(df, scenario_type, run_dir)
        print(f'  stats_{scenario_type}.png')

    print('\n' + '='*60)
    print('AGGREGATION COMPLETE')
    print('='*60)
    print(f'\nResults saved to: {run_dir}')


if __name__ == '__main__':
    main()
