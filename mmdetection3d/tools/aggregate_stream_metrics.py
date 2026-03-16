"""
Aggregate metrics from streamed SMERF predictions.

This script loads streamed prediction pkl files, evaluates them with the dataset,
and generates scenario-based metric aggregations and visualizations.
Supports memory-efficient lazy loading with LRU caching.
"""
import argparse
import os
import os.path as osp
from collections import OrderedDict, defaultdict, OrderedDict as LRU
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mmcv
import torch

matplotlib.use('Agg')

from mmcv import Config, DictAction
from mmdet3d.datasets import build_dataset

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    from functools import lru_cache
except ImportError:
    lru_cache = None


# ===================================================
# Configuration & Constants
# ===================================================

METRIC_COLS_EVAL = ['OpenLane-V2 Score', 'DET_l', 'DET_t', 'TOP_ll', 'TOP_lt']
METRIC_LABELS = {
    'OpenLane-V2 Score': 'OLV2 Score',
    'DET_l': 'DET_l',
    'DET_t': 'DET_t',
    'TOP_ll': 'TOP_ll',
    'TOP_lt': 'TOP_lt',
}
SCENARIO_TITLES = {
    'curvature': 'Road Curvature',
    'lighting': 'Lighting Condition',
    'occlusion': 'Occlusion Level',
    'topology_complexity': 'Topology Complexity',
}


# ===================================================
# Memory & Caching Utilities
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

    # Fallback: use os.sysconf on Unix
    try:
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_AVPHYS_PAGES')
    except (AttributeError, ValueError):
        pass

    return None


def _get_auto_preload_limit_bytes(max_preload_gb):
    """Determine optimal preload limit based on available memory."""
    if max_preload_gb is not None:
        return int(max_preload_gb * (1024 ** 3))

    available_bytes = _get_available_memory_bytes()
    if available_bytes is None:
        # Conservative default: 2GB
        return int(2.0 * (1024 ** 3))

    # Keep cache well below available RAM (use 35% max)
    return max(int(available_bytes * 0.35), int(1.0 * (1024 ** 3)))


class _LazyPklList:
    """List-like object that loads .pkl files on demand with LRU caching."""

    def __init__(self, paths, cache_items=128):
        """
        Initialize lazy list.
        
        Args:
            paths: List of pickle file paths
            cache_items: Number of items to keep in LRU cache (0 disables)
        """
        self.paths = paths
        self.cache_items = cache_items
        self._cache = OrderedDict() if cache_items > 0 else None

    def _load_file(self, idx):
        """Load pickle file at index."""
        return mmcv.load(self.paths[idx])

    def _cache_get(self, idx):
        """Get item from cache, updating LRU order."""
        if self._cache is None:
            return None
        if idx in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(idx)
            return self._cache[idx]
        return None

    def _cache_put(self, idx, item):
        """Put item in cache, evicting oldest if needed."""
        if self._cache is None:
            return
        
        self._cache[idx] = item
        self._cache.move_to_end(idx)
        
        # Evict oldest items if cache full
        while len(self._cache) > self.cache_items:
            self._cache.popitem(last=False)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        """Get item by index with caching."""
        # Try cache first
        if self._cache is not None:
            cached = self._cache_get(idx)
            if cached is not None:
                return cached

        # Load from disk
        item = self._load_file(idx)
        
        # Store in cache
        if self._cache is not None:
            self._cache_put(idx, item)
        
        return item

    def __iter__(self):
        """Iterate over all items."""
        for i in range(len(self)):
            yield self[i]


def _dataset_has_annotations(dataset):
    """Check if dataset contains GT annotations."""
    if not hasattr(dataset, 'data_infos') or not dataset.data_infos:
        return False
    first = dataset.data_infos[0]
    return isinstance(first, dict) and 'annotation' in first


def parse_args():
    parser = argparse.ArgumentParser(
        description='Aggregate metrics from streamed SMERF predictions'
    )
    parser.add_argument('config', help='Config file path')
    parser.add_argument(
        '--stream-dir',
        default='work_dirs/results/stream_outputs',
        help='Directory containing streamed .pkl outputs'
    )
    parser.add_argument(
        '--out-dir',
        default='work_dirs/results',
        help='Directory where aggregated CSVs will be written'
    )
    parser.add_argument(
        '--split',
        type=str,
        default=None,
        help='Override data.test.split in config'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=-1,
        help='Limit number of samples loaded. -1 means all.'
    )
    parser.add_argument(
        '--cache-mode',
        choices=['auto', 'memory', 'lazy'],
        default='auto',
        help=(
            "How to access streamed prediction files. "
            "'memory' preloads once for fastest repeated evaluation, "
            "'lazy' loads on demand with lower RAM use, and "
            "'auto' chooses based on total pickle size."
        ),
    )
    parser.add_argument(
        '--max-preload-gb',
        type=float,
        default=None,
        help=(
            "Optional hard cap for auto-mode preloading in GB. "
            "If omitted, a conservative fraction of available RAM is used."
        ),
    )
    parser.add_argument(
        '--lazy-cache-items',
        type=int,
        default=128,
        help=(
            "Number of predictions kept in an in-memory LRU cache when using lazy mode. "
            "Set 0 to disable caching."
        ),
    )
    parser.add_argument(
        '--eval',
        nargs='+',
        default=None,
        help='Override evaluation metrics list'
    )
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


def _sanitize_eval_kwargs(eval_kwargs):
    """Remove EvalHook-only keys that dataset.evaluate doesn't accept."""
    cleaned = dict(eval_kwargs) if eval_kwargs is not None else {}
    for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
        cleaned.pop(key, None)
    return cleaned


def _get_sample_scenario_labels(info):
    """Return {scenario_type: category_label} for a sample's scenario_meta."""
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


class _ProgressBar:
    """Simple progress bar for single-threaded tasks."""
    def __init__(self, total, desc=''):
        self.total = total
        self.desc = desc
        self.current = 0
        if tqdm is not None:
            self.pbar = tqdm(total=total, desc=desc)
        else:
            self.pbar = None

    def update(self, n=1):
        self.current += n
        if self.pbar is not None:
            self.pbar.update(n)
        else:
            print(f'\r{self.desc} [{self.current}/{self.total}]', end='', flush=True)

    def close(self):
        if self.pbar is not None:
            self.pbar.close()
        else:
            print()


def _load_streamed_prediction_paths(stream_dir):
    """Load all .pkl files from stream_dir."""
    manifest = osp.join(stream_dir, 'manifest.txt')
    paths = []

    # Try manifest first
    if osp.isfile(manifest):
        with open(manifest, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    paths.append(osp.join(stream_dir, line))

    # Fallback: glob all pkl files
    if not paths:
        from glob import glob
        paths = sorted(glob(osp.join(stream_dir, '*.pkl')))

    if not paths:
        raise ValueError(f'No prediction pkl files found in {stream_dir}')

    return paths


def _build_scenarios(dataset):
    """Build scenario index from dataset info."""
    scenarios = {
        'curvature': defaultdict(list),
        'lighting': defaultdict(list),
        'occlusion': defaultdict(list),
        'topology_complexity': defaultdict(list),
    }

    progress = _ProgressBar(len(dataset), 'Scanning scenario labels')
    try:
        for idx, info in enumerate(dataset.data_infos):
            labels = _get_sample_scenario_labels(info)
            for scenario_type, category in labels.items():
                scenarios[scenario_type][category].append(idx)
            progress.update()
    finally:
        progress.close()

    # Remove empty scenario types
    scenarios = {k: dict(v) for k, v in scenarios.items() if v}
    return scenarios


def _evaluate_subset(dataset, outputs, indices, eval_kwargs):
    """Evaluate a subset of outputs by temporarily slicing data_infos."""
    original_data_infos = dataset.data_infos
    try:
        dataset.data_infos = [original_data_infos[i] for i in indices]
        subset_outputs = [outputs[i] for i in indices]
        metrics = dataset.evaluate(subset_outputs, **eval_kwargs)
        return metrics
    finally:
        dataset.data_infos = original_data_infos


def _plot_global_metrics(global_row, run_dir):
    """Plot global metrics as a bar chart."""
    metric_values = [global_row.get(metric) for metric in METRIC_COLS_EVAL]
    labels = [METRIC_LABELS[metric] for metric in METRIC_COLS_EVAL]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(labels, metric_values, color='#2f6db2')
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('Score')
    ax.set_title('Global Metrics')
    ax.grid(axis='y', alpha=0.25)

    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)

    fig.tight_layout()
    fig.savefig(osp.join(run_dir, 'global_metrics.png'), dpi=160)
    plt.close(fig)


def _plot_scenario_table(scenario_type, df, run_dir):
    """Plot metrics for a single scenario type."""
    if df.empty:
        return

    plot_df = df[METRIC_COLS_EVAL].copy()
    labels = [METRIC_LABELS[metric] for metric in METRIC_COLS_EVAL]
    categories = plot_df.index.tolist()
    x_positions = list(range(len(categories)))
    width = 0.8 / max(len(METRIC_COLS_EVAL), 1)

    fig_width = max(8, len(categories) * 1.6)
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))

    for metric_idx, metric in enumerate(METRIC_COLS_EVAL):
        offset = width * (metric_idx - (len(METRIC_COLS_EVAL) - 1) / 2)
        values = plot_df[metric].values
        ax.bar([x + offset for x in x_positions], values, width, label=labels[metric_idx])

    ax.set_xticks(x_positions)
    ax.set_xticklabels(categories, rotation=20, ha='right')
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('Score')
    ax.set_title(SCENARIO_TITLES.get(scenario_type, scenario_type.replace('_', ' ').title()))
    ax.grid(axis='y', alpha=0.25)
    ax.legend(ncol=min(3, len(labels)), frameon=False)

    fig.tight_layout()
    fig.savefig(osp.join(run_dir, f'scenario_{scenario_type}.png'), dpi=160)
    plt.close(fig)


def _save_metric_plots(global_row, scenario_tables, run_dir):
    """Generate and save all metric plots."""
    _plot_global_metrics(global_row, run_dir)
    for scenario_type, df in scenario_tables.items():
        _plot_scenario_table(scenario_type, df, run_dir)


def _choose_output_container(paths, cache_mode, max_preload_gb, lazy_cache_items):
    """Choose between preloading all or lazy loading based on cache_mode."""
    total_bytes = sum(osp.getsize(path) for path in paths)
    auto_limit_bytes = _get_auto_preload_limit_bytes(max_preload_gb)
    
    print(
        f"Discovered {len(paths)} prediction files "
        f"({_format_bytes(total_bytes)} total)."
    )

    if cache_mode == 'memory':
        print('Using memory preloading (will load all predictions into RAM).')
        return 'memory'
    elif cache_mode == 'lazy':
        print('Using lazy loading with LRU cache.')
        return 'lazy'
    else:  # auto mode
        should_preload = total_bytes <= auto_limit_bytes
        
        if should_preload:
            print(
                f'Total size ({_format_bytes(total_bytes)}) is within auto preload '
                f'threshold ({_format_bytes(auto_limit_bytes)}). Preloading all predictions.'
            )
            return 'memory'

        print(
            f"Total prediction size ({_format_bytes(total_bytes)}) exceeds auto preload threshold "
            f"({_format_bytes(auto_limit_bytes)}). Using lazy loading with LRU cache."
        )
        if lazy_cache_items > 0:
            print(f'LRU cache will keep {lazy_cache_items} predictions in memory.')
        else:
            print('Warning: LRU cache disabled (--lazy-cache-items 0). Each item loaded from disk.')
        return 'lazy'


def evaluate_by_scenario(dataset, outputs, eval_kwargs, out_dir):
    """Evaluate metrics globally and by scenario."""
    print('\n' + '='*50)
    print('EVALUATING METRICS BY SCENARIO')
    print('='*50)

    # Global evaluation
    print('\nEvaluating global metrics...')
    global_metrics = dataset.evaluate(outputs, **eval_kwargs)
    
    global_row = {'scenario_type': 'global', 'category': 'all', 'samples': len(outputs)}
    for metric in METRIC_COLS_EVAL:
        global_row[metric] = float(global_metrics.get(metric, 0))
    
    print(f"Global OLV2 Score: {global_row.get('OpenLane-V2 Score', 0):.4f}")

    # Scenario-based evaluation
    print('\nBuilding scenario index...')
    scenarios = _build_scenarios(dataset)
    
    scenario_tables = {}
    all_rows = []

    for scenario_type in sorted(scenarios.keys()):
        print(f'\nEvaluating {scenario_type}...')
        categories = scenarios[scenario_type]
        category_metrics = {}

        for category in sorted(categories.keys()):
            indices = categories[category]
            print(f"  {category} (n={len(indices)})")

            metrics = _evaluate_subset(dataset, outputs, indices, eval_kwargs)
            category_metrics[category] = metrics

            for metric_name, value in metrics.items():
                all_rows.append({
                    'scenario_type': scenario_type,
                    'category': category,
                    'samples': len(indices),
                    'metric': metric_name,
                    'value': float(value)
                })

        # Create dataframe
        if category_metrics:
            rows = []
            for category, metrics in category_metrics.items():
                row = {'category': category, 'samples': len(categories[category])}
                for metric in METRIC_COLS_EVAL:
                    row[metric] = float(metrics.get(metric, 0))
                rows.append(row)
            
            df = pd.DataFrame(rows).set_index('category')
            scenario_tables[scenario_type] = df

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = osp.join(out_dir, f'eval_{timestamp}')
    mmcv.mkdir_or_exist(run_dir)

    # Save global metrics
    global_df = pd.DataFrame([global_row]).set_index('scenario_type')
    global_df.to_csv(osp.join(run_dir, 'global_metrics.csv'))

    # Save scenario results
    for scenario_type, df in scenario_tables.items():
        df.to_csv(osp.join(run_dir, f'scenario_{scenario_type}.csv'))

    # Save combined results
    if all_rows:
        combined_df = pd.DataFrame(all_rows)
        combined_df.to_csv(osp.join(run_dir, 'all_metrics.csv'), index=False)

    # Save metric plots
    _save_metric_plots(global_row, scenario_tables, run_dir)

    print('\n' + '='*50)
    print('EVALUATION COMPLETE')
    print('='*50)
    print(f'Results saved to: {run_dir}')
    print('  global_metrics.csv')
    print('  scenario_*.csv')
    print('  all_metrics.csv')
    print('  *.png (visualizations)')

    return run_dir


def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    # Override split if provided
    if args.split:
        cfg.data.test.split = args.split

    # Setup evaluation
    eval_kwargs = _sanitize_eval_kwargs(cfg.get('evaluation', {}).copy())
    if args.eval is not None:
        eval_kwargs['metrics'] = args.eval

    # Build dataset
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)

    # Load streamed predictions
    print('\nLoading streamed predictions...')
    pkl_paths = _load_streamed_prediction_paths(args.stream_dir)
    print(f'Found {len(pkl_paths)} prediction pkl files')

    # Limit if needed
    if args.max_samples > 0:
        pkl_paths = pkl_paths[:args.max_samples]

    # Choose container mode
    print()
    mode = _choose_output_container(
        pkl_paths, 
        args.cache_mode, 
        args.max_preload_gb,
        args.lazy_cache_items
    )

    # Load predictions
    print('\nLoading predictions...')
    if mode == 'memory':
        # Preload all into memory
        outputs = []
        progress = _ProgressBar(len(pkl_paths), 'Preloading predictions')
        try:
            for pkl_path in pkl_paths:
                output = mmcv.load(pkl_path)
                outputs.append(output)
                progress.update()
        finally:
            progress.close()
    else:
        # Use lazy loading with caching
        outputs = _LazyPklList(pkl_paths, cache_items=args.lazy_cache_items)
        print(f'Lazy loading initialized with {len(pkl_paths)} predictions')

    print(f'Loaded {len(outputs)} predictions')

    # Evaluate
    mmcv.mkdir_or_exist(args.out_dir)
    evaluate_by_scenario(dataset, outputs, eval_kwargs, args.out_dir)


if __name__ == '__main__':
    main()
