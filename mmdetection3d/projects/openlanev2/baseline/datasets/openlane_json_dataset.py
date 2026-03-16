# ==============================================================================
# OpenLane-V2 JSON Dataset Loader
# Direct JSON-based data loading without Collection abstraction
# ==============================================================================

import os
import copy
import glob
import gc
import json
import concurrent.futures

import mmcv
import numpy as np
from pyquaternion import Quaternion

from mmdet.datasets import DATASETS, PIPELINES
from mmdet.datasets.pipelines import Compose
from mmcv.utils import build_from_cfg
from mmdet3d.datasets import Custom3DDataset

from . import decoder


def load_openlane_json(root_dir, split):
    """Load OpenLane-V2 data directly from JSON files.
    
    Args:
        root_dir: Root directory containing split folders
        split: Data split (train, val, test)
        
    Returns:
        List of data_infos with keys: segment_id, timestamp, image_dir, json_path,
                                      sensor, pose, annotation, scenario_meta
    """
    data_infos = []
    split_dir = os.path.join(root_dir, split)

    if not os.path.exists(split_dir):
        print(f"Warning: Split directory {split_dir} not found")
        return data_infos

    segments = sorted(glob.glob(os.path.join(split_dir, "*")))

    for seg in segments:
        seg_id = os.path.basename(seg)
        info_dir = os.path.join(seg, "info")
        image_dir = os.path.join(seg, "image")

        json_files = sorted(glob.glob(info_dir + "/*.json"))

        for jf in json_files:
            with open(jf) as f:
                data = json.load(f)

            timestamp = os.path.basename(jf).replace(".json", "")

            info = {}
            info["segment_id"] = seg_id
            info["timestamp"] = timestamp
            info["image_dir"] = image_dir
            info["json_path"] = jf
            
            # Store raw JSON data
            info["sensor"] = data.get("sensor", {})
            info["pose"] = data.get("pose", {})

            # Store annotations
            if "annotation" in data:
                info["annotation"] = data["annotation"]
            
            # Store scenario tags
            if "scenario_meta" in data:
                info["scenario_meta"] = data["scenario_meta"]

            data_infos.append(info)

    return data_infos


def load_openlane_json_lazy(root_dir, split):
    """Lazy load: collect only file paths without reading JSON contents.

    Each entry is a lightweight stub with keys:
        segment_id, timestamp, json_path, image_dir, _lazy (=True)
    The actual JSON is loaded on first access via _ensure_info_loaded.
    """
    data_infos = []
    split_dir = os.path.join(root_dir, split)

    if not os.path.exists(split_dir):
        print(f"Warning: Split directory {split_dir} not found")
        return data_infos

    segments = sorted(glob.glob(os.path.join(split_dir, "*")))
    
    for seg in segments:
        seg_id = os.path.basename(seg)
        info_dir = os.path.join(seg, "info")
        image_dir = os.path.join(seg, "image")
        json_files = sorted(glob.glob(info_dir + "/*.json"))
        
        for jf in json_files:
            timestamp = os.path.basename(jf).replace(".json", "")
            data_infos.append({
                "segment_id": seg_id,
                "timestamp": timestamp,
                "json_path": jf,
                "image_dir": image_dir,
                "_lazy": True,
            })
    
    return data_infos


def _load_annotation_from_json(json_path):
    """Load annotation from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as handle:
        data = json.load(handle)
    return data.get('annotation')


@DATASETS.register_module()
class OpenLaneJSONDataset(Custom3DDataset):
    """OpenLane-V2 Dataset with direct JSON loading."""

    CLASSES = ("centerline",)

    def __init__(self,
                 data_root,
                 split="train",
                 pipeline=None,
                 test_mode=False,
                 modality=None,
                 lazy_load=False,
                 decoding_function=dict(type='bezier_prediction_decode', 
                                       method_para=dict(n_points=11)),
                 **kwargs):

        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.lazy_load = lazy_load
        self.modality = modality if modality is not None else dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False
        )
        
        # Setup decoding function for predicted lane centerlines
        self.decoding_function = getattr(decoder, decoding_function['type'])
        self.decoding_para = decoding_function['method_para']

        # Load data infos — lazy mode only collects paths, no JSON I/O
        if lazy_load:
            self.data_infos = load_openlane_json_lazy(data_root, split)
        else:
            self.data_infos = load_openlane_json(data_root, split)
        
        # Build pipeline from config
        if pipeline is not None:
            if isinstance(pipeline, list):
                self.pipeline = Compose([build_from_cfg(p, PIPELINES) for p in pipeline])
            else:
                self.pipeline = pipeline
        else:
            self.pipeline = None

    def __len__(self):
        return len(self.data_infos)
    
    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_metas'] = {}
    
    @staticmethod
    def _ensure_info_loaded(info):
        """If info is a lazy stub, read its JSON and fill in full fields."""
        if not info.get('_lazy', False):
            return info
        
        with open(info['json_path']) as f:
            data = json.load(f)
        
        info['sensor'] = data.get('sensor', {})
        info['pose'] = data.get('pose', {})
        if 'annotation' in data:
            info['annotation'] = data['annotation']
        if 'scenario_meta' in data:
            info['scenario_meta'] = data['scenario_meta']
        info['_lazy'] = False
        return info

    @staticmethod
    def _prepare_eval_annotation(annotation):
        """Prepare annotation for evaluation."""
        prepared = copy.deepcopy(annotation)
        
        for lane in prepared['lane_centerline']:
            lane['points'] = np.array(lane['points'], dtype=np.float32)
            # Downsample if 201 points (standard)
            if len(lane['points']) == 201:
                lane['points'] = lane['points'][::20]
        
        for te in prepared['traffic_element']:
            te['points'] = np.array(te['points'], dtype=np.float32)
        
        prepared['topology_lclc'] = np.array(prepared['topology_lclc'], dtype=np.float32)
        prepared['topology_lcte'] = np.array(prepared['topology_lcte'], dtype=np.float32)
        
        return prepared

    def _ensure_eval_annotation_loaded(self, index):
        """Ensure evaluation annotation is prepared and cached."""
        info = self.data_infos[index]
        cached = info.get('_eval_annotation')
        if cached is not None:
            return cached

        annotation = info.get('annotation')
        if annotation is None:
            annotation = _load_annotation_from_json(info['json_path'])

        prepared = self._prepare_eval_annotation(annotation)
        info = dict(info)
        info['_eval_annotation'] = prepared
        self.data_infos[index] = info
        return prepared

    def __getitem__(self, idx):
        """Get item from dataset."""
        # Materialize lazy stubs on first access
        self.data_infos[idx] = self._ensure_info_loaded(self.data_infos[idx])
        data = self.get_data_info(idx)
        if data is None:
            return None
        self.pre_pipeline(data)
        if self.pipeline is not None:
            data = self.pipeline(data)
        return data

    def get_data_info(self, index):
        """Get data info according to index.
        
        Args:
            index (int): Index of the sample data to get.
            
        Returns:
            dict: Data information for pipeline processing.
        """
        info = self.data_infos[index]
        
        input_dict = dict(
            sample_idx=info['timestamp'],
            scene_token=info['segment_id']
        )
        
        # Process camera data
        if self.modality['use_camera'] and 'sensor' in info:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            rots = []
            trans = []
            cam2imgs = []
            
            for cam_name, cam_info in info['sensor'].items():
                if 'image_path' not in cam_info:
                    continue
                    
                image_path = cam_info['image_path']
                image_paths.append(os.path.join(self.data_root, image_path))
                
                # Get extrinsic and intrinsic matrices
                if 'extrinsic' in cam_info and 'intrinsic' in cam_info:
                    extrinsic = cam_info['extrinsic']
                    intrinsic = cam_info['intrinsic']
                    
                    # Compute lidar to camera rotation and translation
                    lidar2cam_r = np.linalg.inv(np.array(extrinsic['rotation']))
                    lidar2cam_t = cam_info['extrinsic']['translation'] @ lidar2cam_r.T
                    
                    # Build 4x4 lidar to camera matrix
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t
                    
                    # Get intrinsic matrix
                    intrinsic_matrix = np.array(intrinsic['K'])
                    
                    # Build view padding matrix (4x4)
                    viewpad = np.eye(4)
                    viewpad[:intrinsic_matrix.shape[0], :intrinsic_matrix.shape[1]] = intrinsic_matrix
                    
                    # Compute lidar to image projection matrix
                    lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                    
                    lidar2img_rts.append(lidar2img_rt)
                    cam_intrinsics.append(viewpad)
                    lidar2cam_rts.append(lidar2cam_rt.T)
                    
                    # Also compute rots, trans, and cam2imgs for pipeline compatibility
                    rots.append(np.linalg.inv(np.array(extrinsic['rotation'])))
                    trans.append(-np.array(cam_info['extrinsic']['translation']))
                    cam2imgs.append(intrinsic_matrix)
            
            if image_paths:
                input_dict.update(
                    dict(
                        img_paths=image_paths,
                        lidar2img=lidar2img_rts,
                        cam_intrinsic=cam_intrinsics,
                        lidar2cam=lidar2cam_rts,
                        rots=rots,
                        trans=trans,
                        cam2imgs=cam2imgs,
                    ))
        
        # Process pose/can_bus data
        if 'pose' in info:
            pose = info['pose']
            can_bus = np.zeros(18)
            
            if 'translation' in pose:
                can_bus[:3] = np.array(pose['translation'])
            
            if 'rotation' in pose:
                input_dict['lidar2global_rotation'] = np.array(pose['rotation'])
                try:
                    rotation = Quaternion._from_matrix(np.array(pose['rotation']))
                    can_bus[3:7] = rotation
                    patch_angle = rotation.yaw_pitch_roll[0] / np.pi * 180
                    if patch_angle < 0:
                        patch_angle += 360
                    can_bus[-2] = patch_angle / 180 * np.pi
                    can_bus[-1] = patch_angle
                except:
                    pass
            
            input_dict['can_bus'] = can_bus
        
        # Add annotations if available
        if "annotation" in info:
            input_dict["ann_info"] = info["annotation"]
        
        # Add scenario tags if available
        if "scenario_meta" in info:
            input_dict["scenario_meta"] = info["scenario_meta"]
        
        # Add SD map (spatial-data map for map encoding)
        # Get from sensor data if available, otherwise provide empty map with required categories
        sd_map = {}
        if 'sensor' in info and 'sd_map' in info['sensor']:
            sd_map = info['sensor']['sd_map']
        else:
            # Provide empty SD map with required categories for template consistency
            sd_map = {
                'road': [],
                'cross_walk': [],
                'side_walk': []
            }
        input_dict['sd_map'] = sd_map
        
        return input_dict

    def format_openlanev2_gt(self):
        """Prepare ground truth annotations for evaluation."""
        total = len(self.data_infos)
        print(f'Preparing GT annotations for {total} samples ...')

        pending_indices = [
            idx for idx, info in enumerate(self.data_infos)
            if info.get('_eval_annotation') is None
        ]

        if pending_indices:
            workers = min(max(4, (os.cpu_count() or 4)), 16, len(pending_indices))
            print(f'Loading and converting {len(pending_indices)} annotations with {workers} workers ...')
            load_progress = mmcv.ProgressBar(len(pending_indices))

            def _build_annotation(index):
                info = self.data_infos[index]
                annotation = info.get('annotation')
                if annotation is None:
                    annotation = _load_annotation_from_json(info['json_path'])
                return index, self._prepare_eval_annotation(annotation)

            if workers <= 1:
                for idx in pending_indices:
                    out_idx, prepared = _build_annotation(idx)
                    info = dict(self.data_infos[out_idx])
                    info['_eval_annotation'] = prepared
                    self.data_infos[out_idx] = info
                    load_progress.update()
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                    future_to_idx = {executor.submit(_build_annotation, idx): idx for idx in pending_indices}
                    for future in concurrent.futures.as_completed(future_to_idx):
                        out_idx, prepared = future.result()
                        info = dict(self.data_infos[out_idx])
                        info['_eval_annotation'] = prepared
                        self.data_infos[out_idx] = info
                        load_progress.update()

        gt_dict = {}
        progress = mmcv.ProgressBar(total)
        for idx in range(total):
            info = self.data_infos[idx]
            key = (self.split, info['segment_id'], str(info['timestamp']))
            gt_dict[key] = {'annotation': self._ensure_eval_annotation_loaded(idx)}
            progress.update()
        
        return gt_dict

    def format_results(self, results):
        """Format predictions into OpenLane-V2 format."""
        pred_dict = {}
        pred_dict['method'] = 'SMERF'
        pred_dict['authors'] = []
        pred_dict['e-mail'] = 'dummy'
        pred_dict['institution / company'] = 'OpenDriveLab'
        pred_dict['country / region'] = 'CN'
        pred_dict['results'] = {}
        
        for idx, result in enumerate(results):
            info = self.data_infos[idx]
            key = (self.split, info['segment_id'], str(info['timestamp']))

            pred_info = dict(
                lane_centerline=[],
                traffic_element=[],
                topology_lclc=None,
                topology_lcte=None
            )

            valid_indices = None
            if result.get('lane_results') is not None:
                lane_results = result['lane_results']
                scores = lane_results[1]
                valid_indices = np.argsort(-scores)
                lanes = lane_results[0][valid_indices]
                lanes = lanes.reshape(-1, lanes.shape[-1] // 3, 3)
                scores = scores[valid_indices]
                
                for pred_idx, (lane, score) in enumerate(zip(lanes, scores)):
                    # Decode lane if needed (e.g., from Bezier coefficients)
                    if hasattr(self, 'decoding_function'):
                        lane = self.decoding_function(np.array([lane]), **self.decoding_para)[0]
                    
                    lc_info = dict(
                        id=10000 + pred_idx,
                        points=lane.astype(np.float32),
                        confidence=float(score)
                    )
                    pred_info['lane_centerline'].append(lc_info)

            te_valid_indices = None
            if result.get('bbox_results') is not None:
                te_results = result['bbox_results']
                scores = te_results[1]
                te_valid_indices = np.argsort(-scores)
                tes = te_results[0][te_valid_indices]
                scores = scores[te_valid_indices]
                class_idxs = te_results[2][te_valid_indices]
                
                for pred_idx, (te, score, class_idx) in enumerate(zip(tes, scores, class_idxs)):
                    te_info = dict(
                        id=20000 + pred_idx,
                        category=1 if class_idx < 4 else 2,
                        attribute=int(class_idx),
                        points=te.reshape(2, 2).astype(np.float32),
                        confidence=float(score)
                    )
                    pred_info['traffic_element'].append(te_info)

            if result.get('lclc_results') is not None and valid_indices is not None:
                pred_info['topology_lclc'] = result['lclc_results'].astype(np.float32)[valid_indices][:, valid_indices]
            else:
                pred_info['topology_lclc'] = np.zeros(
                    (len(pred_info['lane_centerline']), len(pred_info['lane_centerline'])),
                    dtype=np.float32
                )

            if result.get('lcte_results') is not None and valid_indices is not None and te_valid_indices is not None:
                pred_info['topology_lcte'] = result['lcte_results'].astype(np.float32)[valid_indices][:, te_valid_indices]
            else:
                pred_info['topology_lcte'] = np.zeros(
                    (len(pred_info['lane_centerline']), len(pred_info['traffic_element'])),
                    dtype=np.float32
                )

            pred_dict['results'][key] = dict(predictions=pred_info)

        return pred_dict

    def evaluate(self, results, logger=None, eval_kwargs=None, **kwargs):
        """Evaluate using OpenLane-V2 metrics."""
        try:
            from openlanev2.utils import format_metric
            from openlanev2.centerline.evaluation import evaluate as openlanev2_evaluate
        except ImportError:
            raise ImportError("OpenLane-V2 evaluation tools not found. Please install openlanev2.")

        if logger is None:
            import logging
            logger = logging.getLogger('mmdet')

        logger.info('Formatting ground truth...')
        gt_dict = self.format_openlanev2_gt()

        logger.info('Formatting predictions...')
        pred_dict = self.format_results(results)

        logger.info('Running OpenLane-V2 evaluation...')
        metric_results = openlanev2_evaluate(gt_dict, pred_dict)
        del gt_dict, pred_dict
        gc.collect()
        
        format_metric(metric_results)
        
        metric_results = {
            'OpenLane-V2 Score': metric_results['OpenLane-V2 Score']['score'],
            'DET_l': metric_results['OpenLane-V2 Score']['DET_l'],
            'DET_t': metric_results['OpenLane-V2 Score']['DET_t'],
            'TOP_ll': metric_results['OpenLane-V2 Score']['TOP_ll'],
            'TOP_lt': metric_results['OpenLane-V2 Score']['TOP_lt'],
        }
        
        return metric_results
