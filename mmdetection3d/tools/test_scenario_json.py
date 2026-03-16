"""
Test and evaluate SMERF with OpenLane-V2 JSON dataset.
Supports scenario-based evaluation and metric visualization.
"""

import argparse
import glob
import os
import os.path as osp
import warnings
from datetime import datetime

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mmcv
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmcv.utils import get_logger

from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model


# ---------------------------------------------------
# Visualization Helpers
# ---------------------------------------------------

METRIC_LABELS = {
    'OpenLane-V2 Score': 'OLV2 Score',
    'DET_l': 'DET$_l$',
    'DET_t': 'DET$_t$',
    'TOP_ll': 'TOP$_{ll}$',
    'TOP_lt': 'TOP$_{lt}$',
}
METRIC_COLS = list(METRIC_LABELS.keys())
SCENARIO_TITLES = {
    'curvature': 'Road Curvature',
    'lighting': 'Lighting Condition',
    'occlusion': 'Occlusion Level',
    'topology_complexity': 'Topology Complexity',
}
PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
METRIC_COLS_EVAL = ['OpenLane-V2 Score', 'DET_l', 'DET_t', 'TOP_ll', 'TOP_lt']


def _bar_chart_per_scenario(df, scenario_type, global_score, run_dir):
    """Grouped bar chart: all 5 metrics per category, with global baseline line."""
    if df.empty:
        return
        
    categories = df.index.tolist()
    n_cats = len(categories)
    n_metrics = len(METRIC_COLS)
    x = np.arange(n_cats)
    width = 0.15

    fig, ax = plt.subplots(figsize=(max(7, n_cats * 2.2), 5))
    for j, metric in enumerate(METRIC_COLS):
        if metric not in df.columns:
            continue
        vals = df[metric].values.astype(float)
        bars = ax.bar(x + j * width, vals, width, label=METRIC_LABELS[metric],
                      color=PALETTE[j], edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7)

    ax.axhline(y=global_score, color='grey', linestyle='--', linewidth=1,
               label=f'Global OLV2 ({global_score:.3f})')
    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(f'Scenario: {SCENARIO_TITLES.get(scenario_type, scenario_type)}',
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
    """Horizontal bar chart showing OLV2 Score delta from global for every category."""
    rows = []
    for stype, df in scenario_tables.items():
        for cat in df.index:
            delta = float(df.loc[cat, 'OpenLane-V2 Score']) - global_score
            rows.append({'scenario': SCENARIO_TITLES.get(stype, stype),
                         'category': cat, 'delta': delta,
                         'samples': int(df.loc[cat, 'samples'])})
    if not rows:
        return
    delta_df = pd.DataFrame(rows).sort_values('delta')

    fig, ax = plt.subplots(figsize=(8, max(4, len(rows) * 0.55)))
    colors = ['#C44E52' if d < 0 else '#55A868' for d in delta_df['delta']]
    labels = [f"{r['category']}  (n={r['samples']})" for _, r in delta_df.iterrows()]
    bars = ax.barh(range(len(delta_df)), delta_df['delta'], color=colors,
                   edgecolor='white', height=0.6)
    ax.set_yticks(range(len(delta_df)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(x=0, color='grey', linewidth=0.8)
    ax.set_xlabel('$\\Delta$ OLV2 Score (vs. global)', fontsize=11)
    ax.set_title('Performance Gap by Scenario Category', fontsize=13, fontweight='bold')
    for bar, v in zip(bars, delta_df['delta']):
        ax.text(v + (0.003 if v >= 0 else -0.003),
                bar.get_y() + bar.get_height() / 2,
                f'{v:+.3f}', ha='left' if v >= 0 else 'right',
                va='center', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(osp.join(run_dir, 'score_delta.png'), dpi=200)
    fig.savefig(osp.join(run_dir, 'score_delta.pdf'))
    plt.close(fig)


def _heatmap(scenario_tables, run_dir):
    """Heatmap: rows = all categories across scenario types, cols = metrics."""
    rows = []
    labels = []
    for stype, df in scenario_tables.items():
        prefix = SCENARIO_TITLES.get(stype, stype)
        for cat in df.index:
            labels.append(f'{prefix} / {cat}')
            row_data = [df.loc[cat, m] if m in df.columns else 0 for m in METRIC_COLS]
            rows.append(row_data)
    
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
            ax.text(j, i, f'{mat[i, j]:.3f}', ha='center', va='center',
                    fontsize=8, color='white' if mat[i, j] < 0.2 else 'black')
    ax.set_title('Metric Heatmap Across All Scenarios', fontsize=13, fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.8, label='Score')
    fig.tight_layout()
    fig.savefig(osp.join(run_dir, 'heatmap.png'), dpi=200)
    fig.savefig(osp.join(run_dir, 'heatmap.pdf'))
    plt.close(fig)


def _sample_distribution(scenario_tables, run_dir):
    """Pie charts showing sample counts per category for each scenario type."""
    n_types = len(scenario_tables)
    if n_types == 0:
        return
    fig, axes = plt.subplots(1, n_types, figsize=(4.5 * n_types, 4))
    if n_types == 1:
        axes = [axes]
    for ax, (stype, df) in zip(axes, scenario_tables.items()):
        counts = df['samples'].astype(int)
        cat_labels = [f'{cat}\n(n={int(c)})' for cat, c in zip(df.index, counts)]
        ax.pie(counts, labels=cat_labels, autopct='%1.0f%%', startangle=90,
               colors=PALETTE[:len(counts)], textprops={'fontsize': 8})
        ax.set_title(SCENARIO_TITLES.get(stype, stype), fontsize=11, fontweight='bold')
    fig.suptitle('Sample Distribution by Scenario', fontsize=13,
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(osp.join(run_dir, 'sample_distribution.png'), dpi=200,
                bbox_inches='tight')
    fig.savefig(osp.join(run_dir, 'sample_distribution.pdf'),
                bbox_inches='tight')
    plt.close(fig)


def _radar_chart(scenario_tables, global_row, run_dir):
    """One radar chart per scenario type overlaying all categories + global."""
    angles = np.linspace(0, 2 * np.pi, len(METRIC_COLS), endpoint=False).tolist()
    angles += angles[:1]

    for stype, df in scenario_tables.items():
        if df.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        gvals = [float(global_row.get(m, 0)) for m in METRIC_COLS]
        gvals += gvals[:1]
        ax.plot(angles, gvals, 'k--', linewidth=1.2, label='Global')
        ax.fill(angles, gvals, alpha=0.05, color='grey')
        for idx, cat in enumerate(df.index):
            vals = [float(df.loc[cat, m]) if m in df.columns else 0 for m in METRIC_COLS]
            vals += vals[:1]
            ax.plot(angles, vals, linewidth=1.5,
                    label=f'{cat} (n={int(df.loc[cat, "samples"])})',
                    color=PALETTE[idx % len(PALETTE)])
            ax.fill(angles, vals, alpha=0.08,
                    color=PALETTE[idx % len(PALETTE)])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([METRIC_LABELS[m] for m in METRIC_COLS], fontsize=9)
        ax.set_ylim(0, min(1.0, df[METRIC_COLS].max().max() + 0.15))
        ax.set_title(f'{SCENARIO_TITLES.get(stype, stype)}',
                     fontsize=13, fontweight='bold', pad=20)
        ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.3, 1.1))
        fig.tight_layout()
        fig.savefig(osp.join(run_dir, f'radar_{stype}.png'), dpi=200,
                    bbox_inches='tight')
        fig.savefig(osp.join(run_dir, f'radar_{stype}.pdf'),
                    bbox_inches='tight')
        plt.close(fig)


def generate_visualizations(scenario_tables, global_row, run_dir):
    """Generate all visualization plots and save to run_dir."""
    global_score = float(global_row.get('OpenLane-V2 Score', 0))

    for stype, df in scenario_tables.items():
        _bar_chart_per_scenario(df, stype, global_score, run_dir)

    _score_drop_chart(scenario_tables, global_score, run_dir)
    _heatmap(scenario_tables, run_dir)
    _sample_distribution(scenario_tables, run_dir)
    _radar_chart(scenario_tables, global_row, run_dir)

    print(f"\nVisualizations saved to: {run_dir}")
    print("  bar_<scenario>.png/pdf        : grouped bar charts per scenario")
    print("  score_delta.png/pdf           : performance gap vs. global")
    print("  heatmap.png/pdf               : metric heatmap across all categories")
    print("  sample_distribution.png/pdf   : pie charts of sample counts")
    print("  radar_<scenario>.png/pdf      : radar overlay per scenario")


def parse_args():
    parser = argparse.ArgumentParser(description="SMERF + OpenLane-V2 evaluation")

    parser.add_argument("config", help="config file")
    parser.add_argument("checkpoint", help="checkpoint file")

    parser.add_argument("--out", action="store_true")
    parser.add_argument("--out-dir", default="work_dirs/results")

    parser.add_argument(
        "--eval",
        nargs="+",
        help="evaluation metrics"
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=-1,
                        help="Limit number of samples. Use -1 for full split.")
    parser.add_argument("--workers-per-gpu", type=int, default=0,
                        help="Dataloader workers per GPU (0 is safest on Windows).")
    parser.add_argument("--stream-out", action="store_true",
                        help="Stream per-sample predictions to disk to reduce RAM usage.")
    parser.add_argument("--stream-dir", default="",
                        help="Directory for streamed predictions. Default: <out-dir>/stream_outputs")
    parser.add_argument("--clear-cache-interval", type=int, default=10,
                        help="Call torch.cuda.empty_cache() every N batches. <=0 disables it.")
    parser.add_argument("--split", type=str, default=None,
                        help="Override data.test.split in config (e.g. train, val, test).")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use (0 is default). Use -1 for CPU inference (slower but uses less memory).")
    parser.add_argument("--resume-from-stream", action="store_true",
                        help="Skip inference and load streamed *.pkl predictions from --stream-dir.")

    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction
    )

    args = parser.parse_args()
    return args


def _sanitize_eval_kwargs(eval_kwargs):
    """Remove EvalHook-only keys that dataset.evaluate does not accept."""
    cleaned = dict(eval_kwargs) if eval_kwargs is not None else {}
    for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
        cleaned.pop(key, None)
    return cleaned


def _unpack_data_container(data):
    """Unpack DataContainer objects in data dict for CPU inference."""
    from mmcv.parallel.data_container import DataContainer
    
    unpacked = {}
    for key, val in data.items():
        if isinstance(val, DataContainer):
            item = val.data
            # DataLoader keeps a per-GPU list even when samples_per_gpu == 1.
            # Remove only one batch wrapper level to match model.forward_test inputs.
            if isinstance(item, (list, tuple)) and len(item) == 1:
                item = item[0]
            unpacked[key] = item
        else:
            unpacked[key] = val
    return unpacked


def _load_streamed_outputs(stream_dir):
    """Load per-sample prediction files previously written by --stream-out."""
    if not osp.isdir(stream_dir):
        raise FileNotFoundError(f"Stream directory not found: {stream_dir}")

    manifest_file = osp.join(stream_dir, "manifest.txt")
    if osp.isfile(manifest_file):
        with open(manifest_file, "r", encoding="utf-8") as f:
            files = [line.strip() for line in f if line.strip()]
    else:
        files = sorted(glob.glob(osp.join(stream_dir, "*.pkl")))

    if not files:
        raise RuntimeError(f"No streamed prediction files found in: {stream_dir}")

    outputs = [mmcv.load(fp) for fp in files]
    print(f"Loaded {len(outputs)} streamed predictions from: {stream_dir}")
    return outputs


def _get_sample_scenario_labels(info):
    """Return {scenario_type: category_label} for a single sample's scenario_meta."""
    out = {}
    meta = info.get('scenario_meta', {})

    if 'curvature' in meta:
        v = meta['curvature'].get('value_m_inv')
        t = meta['curvature'].get('thresholds_m_inv', {})
        if v is not None:
            if v <= t.get('straight', 0.003):
                out['curvature'] = 'straight'
            elif v <= t.get('low', 0.008):
                out['curvature'] = 'low curvature'
            elif v <= t.get('medium', 0.02):
                out['curvature'] = 'medium curvature'
            else:
                out['curvature'] = 'high curvature'

    if 'topology_complexity' in meta:
        v = meta['topology_complexity'].get('value')
        if v is not None:
            if v <= 0.3:
                out['topology_complexity'] = 'low topology complexity'
            elif v <= 0.6:
                out['topology_complexity'] = 'medium topology complexity'
            else:
                out['topology_complexity'] = 'high topology complexity'

    if 'lighting' in meta:
        lbl = meta['lighting'].get('label')
        if lbl:
            out['lighting'] = lbl

    if 'occlusion' in meta:
        lbl = meta['occlusion'].get('label')
        if lbl:
            out['occlusion'] = lbl

    return out


def evaluate_by_scenario(dataset, outputs, eval_kwargs, out_dir):
    """Evaluate model performance grouped by scenario attributes."""
    metric_cols = METRIC_COLS_EVAL
    all_rows = []

    # Ensure scenario metadata is available even when dataset.lazy_load=True.
    materialized = 0
    for i in range(len(dataset.data_infos)):
        info = dataset.data_infos[i]
        if info.get('_lazy', False) and hasattr(dataset, '_ensure_info_loaded'):
            dataset.data_infos[i] = dataset._ensure_info_loaded(info)
            materialized += 1
    if materialized > 0:
        print(f"Materialized {materialized} lazy samples for scenario metadata.")

    # ---- Global ----
    print("\n==============================")
    print("GLOBAL METRICS")
    print("==============================")

    global_metrics = dataset.evaluate(outputs, eval_kwargs=eval_kwargs)
    print(global_metrics)

    global_row = {'scenario_type': 'global', 'category': 'all', 'samples': len(outputs)}
    global_row.update({k: global_metrics.get(k) for k in metric_cols})
    all_rows.append(global_row)

    # ---- Build scenario groups ----
    scenarios = {
        "curvature": {},
        "lighting": {},
        "occlusion": {},
        "topology_complexity": {}
    }

    scenario_meta_count = 0

    for i in range(len(dataset)):
        info = dataset.data_infos[i]
        if info.get('scenario_meta'):
            scenario_meta_count += 1
        for stype, label in _get_sample_scenario_labels(info).items():
            scenarios[stype].setdefault(label, []).append(i)

    print(f"Samples with scenario_meta: {scenario_meta_count}/{len(dataset)}")

    # ---- Per-scenario evaluation ----
    print("\n==============================")
    print("SCENARIO BREAKDOWN")
    print("==============================")

    scenario_tables = {}

    for scenario_type in scenarios:
        print("\n---", scenario_type.upper(), "---")

        if len(scenarios[scenario_type]) == 0:
            print(f"  No data available")
            continue

        rows = []
        for category in sorted(scenarios[scenario_type].keys()):
            indices = scenarios[scenario_type][category]
            subset_outputs = [outputs[i] for i in indices]

            original_data_infos = dataset.data_infos
            try:
                dataset.data_infos = [dataset.data_infos[i] for i in indices]
                metrics = dataset.evaluate(subset_outputs, eval_kwargs=eval_kwargs)
            except ValueError as exc:
                # OpenLane-V2 topology eval can be undefined for tiny subsets.
                if 'need at least one array to concatenate' in str(exc):
                    print(f"  {category}: skipped ({len(indices)} samples, invalid topology subset)")
                    continue
                raise
            finally:
                dataset.data_infos = original_data_infos

            row = {'category': category, 'samples': len(indices)}
            row.update({k: metrics.get(k) for k in metric_cols})
            rows.append(row)

            combined_row = {'scenario_type': scenario_type, **row}
            all_rows.append(combined_row)

            print(f"  {category}: {len(indices)} samples")
            print(f"    Metrics: {metrics}")

        df = pd.DataFrame(rows).set_index('category')
        scenario_tables[scenario_type] = df

    # ---- Save everything ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = osp.join(out_dir, f"eval_{timestamp}")
    mmcv.mkdir_or_exist(run_dir)

    # Global metrics
    global_df = pd.DataFrame([global_row]).set_index('scenario_type')
    global_df.to_csv(osp.join(run_dir, "global_metrics.csv"))

    # Per-scenario tables
    for scenario_type, df in scenario_tables.items():
        df.to_csv(osp.join(run_dir, f"scenario_{scenario_type}.csv"))

    # Combined table
    combined_df = pd.DataFrame(all_rows)
    combined_df.to_csv(osp.join(run_dir, "all_metrics.csv"), index=False)

    # Pretty-print summary
    print("\n==============================")
    print("SAVED RESULTS")
    print("==============================")
    print(f"Directory: {run_dir}")
    print(f"\nGlobal:")
    print(global_df.to_string())
    for scenario_type, df in scenario_tables.items():
        print(f"\n{scenario_type}:")
        print(df.to_string())

    # Generate visualizations
    generate_visualizations(scenario_tables, global_row, run_dir)

    return run_dir


def run_inference(model, data_loader, gpu_id=0, out_dir=None, stream_out=False, 
                 stream_dir=None, clear_cache_interval=10):
    """Run inference on dataset."""
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    outputs = []
    streamed_files = []

    if stream_out and stream_dir:
        mmcv.mkdir_or_exist(stream_dir)

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            # MMDataParallel handles scatter/copy to GPU internally.
            # Only unpack DataContainers for non-parallel (CPU) execution.
            if not isinstance(model, MMDataParallel):
                data = _unpack_data_container(data)
            
            result = model(return_loss=False, rescale=True, **data)
            if not isinstance(result, list):
                result = [result]

            outputs.extend(result)

            if stream_out and stream_dir:
                for idx, item in enumerate(result):
                    out_path = osp.join(stream_dir, f"{i * data_loader.batch_size + idx:07d}.pkl")
                    mmcv.dump(item, out_path)
                    streamed_files.append(out_path)

            for _ in range(len(result)):
                prog_bar.update()

            if torch.cuda.is_available() and clear_cache_interval > 0 and (i + 1) % clear_cache_interval == 0:
                torch.cuda.empty_cache()

    return outputs, streamed_files


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    mmcv.mkdir_or_exist(osp.abspath(args.out_dir))
    set_random_seed(args.seed)

    if args.split:
        cfg.data.test.split = args.split
        print(f"Overriding data.test.split = '{args.split}'")

    # ---- Dataset ----
    print("Building dataset...")
    dataset = build_dataset(cfg.data.test)

    if args.max_samples > 0 and len(dataset) > args.max_samples:
        print(f"Limiting to first {args.max_samples} of {len(dataset)} samples")
        dataset.data_infos = dataset.data_infos[:args.max_samples]

    stream_dir = args.stream_dir if args.stream_dir else osp.join(args.out_dir, "stream_outputs")
    effective_stream_out = bool(args.stream_out)

    if args.resume_from_stream:
        print("Resuming from streamed predictions (skipping inference)...")
        outputs = _load_streamed_outputs(stream_dir)
        streamed_files = []

        if len(outputs) != len(dataset):
            if len(outputs) < len(dataset):
                print(f"Aligning dataset to streamed outputs: {len(dataset)} -> {len(outputs)}")
                dataset.data_infos = dataset.data_infos[:len(outputs)]
            else:
                print(f"Aligning outputs to dataset size: {len(outputs)} -> {len(dataset)}")
                outputs = outputs[:len(dataset)]
    else:
        # ---- Model ----
        print("Building model...")
        model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
        checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

        if "CLASSES" in checkpoint.get("meta", {}):
            model.CLASSES = checkpoint["meta"]["CLASSES"]
        else:
            model.CLASSES = dataset.CLASSES

        # ---- DataLoader ----
        print("Building dataloader...")
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=args.workers_per_gpu,
            dist=False,
            shuffle=False,
        )

        # Use CPU or GPU based on --gpu flag
        if args.gpu < 0:
            print("Using CPU inference (slower but memory-efficient for 4GB GPU constraints).")
            model.eval()
        else:
            print(f"Using GPU {args.gpu}")
            model = MMDataParallel(model, device_ids=[args.gpu])

        # ---- Inference ----
        print("Running inference...")
        outputs, streamed_files = run_inference(
            model,
            data_loader,
            gpu_id=args.gpu,
            out_dir=args.out_dir,
            stream_out=effective_stream_out,
            stream_dir=stream_dir,
            clear_cache_interval=args.clear_cache_interval,
        )

    # ---- Save ----
    if args.out and outputs is not None:
        out_file = osp.join(args.out_dir, "results.pkl")
        mmcv.dump(outputs, out_file)
        print("Saved:", out_file)

    if effective_stream_out:
        manifest_file = osp.join(stream_dir, "manifest.txt")
        with open(manifest_file, "w", encoding="utf-8") as f:
            for fp in streamed_files:
                f.write(fp + "\n")
        print(f"Streamed {len(streamed_files)} predictions to: {stream_dir}")

    # ---- Evaluate ----
    if args.eval:
        eval_kwargs = _sanitize_eval_kwargs(cfg.get("evaluation", {}).copy())
        print("Running scenario-based evaluation...")
        evaluate_by_scenario(dataset, outputs, eval_kwargs, args.out_dir)


if __name__ == "__main__":
    main()
