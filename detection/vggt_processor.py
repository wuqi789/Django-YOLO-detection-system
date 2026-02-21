import os
import sys
import cv2
import torch
import numpy as np
import glob
import gc
import time
import shutil
from datetime import datetime
import threading
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VGGT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'vggt')
sys.path.insert(0, VGGT_DIR)

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from visual_util import predictions_to_glb

_vggt_model = None
_vggt_model_lock = threading.Lock()

_progress_store = {}
_progress_lock = threading.Lock()


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    global _vggt_model
    
    with _vggt_model_lock:
        if _vggt_model is None:
            logger.info("Initializing VGGT model...")
            device = get_device()
            
            _vggt_model = VGGT()
            model_path = os.path.join(VGGT_DIR, 'model.pt')
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            logger.info(f"Loading model from: {model_path}")
            _vggt_model.load_state_dict(torch.load(model_path, map_location=device))
            _vggt_model.eval()
            _vggt_model = _vggt_model.to(device)
            logger.info("VGGT model loaded successfully")
        
        return _vggt_model


def init_progress(task_id):
    with _progress_lock:
        _progress_store[task_id] = {
            'status': 'initialized',
            'progress': 0,
            'message': 'Task initialized',
            'glb_path': None,
            'target_dir': None,
            'error': None
        }


def update_progress(task_id, status, progress, message, glb_path=None, target_dir=None, error=None):
    with _progress_lock:
        if task_id in _progress_store:
            _progress_store[task_id].update({
                'status': status,
                'progress': progress,
                'message': message,
                'glb_path': glb_path,
                'target_dir': target_dir,
                'error': error
            })


def get_progress(task_id):
    with _progress_lock:
        return _progress_store.get(task_id, {
            'status': 'not_found',
            'progress': 0,
            'message': 'Task not found',
            'glb_path': None,
            'target_dir': None,
            'error': None
        })


def cleanup_progress(task_id):
    with _progress_lock:
        if task_id in _progress_store:
            del _progress_store[task_id]


def process_uploaded_files(video_path=None, image_paths=None, output_base_dir=None):
    """
    Process uploaded video and/or images, extract frames and save to target directory.
    
    Returns:
        tuple: (target_dir, image_paths_list)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = os.path.join(output_base_dir, f"input_images_{timestamp}")
    target_dir_images = os.path.join(target_dir, "images")
    
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    os.makedirs(target_dir_images)
    
    saved_image_paths = []
    
    if image_paths:
        for img_path in image_paths:
            if os.path.exists(img_path):
                dst_path = os.path.join(target_dir_images, os.path.basename(img_path))
                shutil.copy(img_path, dst_path)
                saved_image_paths.append(dst_path)
    
    if video_path and os.path.exists(video_path):
        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 1) if fps > 0 else 1
        
        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            count += 1
            if count % frame_interval == 0:
                image_path = os.path.join(target_dir_images, f"{video_frame_num:06}.png")
                cv2.imwrite(image_path, frame)
                saved_image_paths.append(image_path)
                video_frame_num += 1
        vs.release()
    
    saved_image_paths = sorted(saved_image_paths)
    
    if len(saved_image_paths) == 0:
        shutil.rmtree(target_dir)
        raise ValueError("No valid images found from uploaded files")
    
    return target_dir, saved_image_paths


def run_vggt_inference(target_dir, model):
    """
    Run VGGT model inference on images in target_dir/images.
    
    Returns:
        dict: Predictions dictionary
    """
    device = get_device()
    
    if device == "cpu":
        logger.warning("CUDA not available, running on CPU (may be slow)")
    
    model = model.to(device)
    model.eval()
    
    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    
    if len(image_names) == 0:
        raise ValueError("No images found in target directory")
    
    logger.info(f"Found {len(image_names)} images for processing")
    
    images = load_and_preprocess_images(image_names).to(device)
    logger.info(f"Preprocessed images shape: {images.shape}")
    
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)
    predictions['pose_enc_list'] = None
    
    depth_map = predictions["depth"]
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points
    
    torch.cuda.empty_cache()
    
    return predictions


def generate_glb(predictions, target_dir, conf_thres=50.0, frame_filter="All", 
                 mask_black_bg=False, mask_white_bg=False, show_cam=True, 
                 mask_sky=False, prediction_mode="Depthmap and Camera Branch"):
    """
    Generate GLB file from predictions.
    
    Returns:
        str: Path to generated GLB file
    """
    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb",
    )
    
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=frame_filter,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        target_dir=target_dir,
        prediction_mode=prediction_mode,
    )
    glbscene.export(file_obj=glbfile)
    
    return glbfile


def process_3d_reconstruction(task_id, video_path=None, image_paths=None, output_base_dir=None,
                              conf_thres=50.0, frame_filter="All", mask_black_bg=False, 
                              mask_white_bg=False, show_cam=True, mask_sky=False, 
                              prediction_mode="Depthmap and Camera Branch"):
    """
    Main function to process 3D reconstruction asynchronously.
    """
    try:
        init_progress(task_id)
        update_progress(task_id, 'processing', 10, 'Loading VGGT model...')
        
        model = load_model()
        
        update_progress(task_id, 'processing', 20, 'Processing uploaded files...')
        target_dir, saved_paths = process_uploaded_files(video_path, image_paths, output_base_dir)
        
        update_progress(task_id, 'processing', 30, f'Running VGGT inference on {len(saved_paths)} images...')
        predictions = run_vggt_inference(target_dir, model)
        
        prediction_save_path = os.path.join(target_dir, "predictions.npz")
        np.savez(prediction_save_path, **predictions)
        
        update_progress(task_id, 'processing', 70, 'Generating 3D model...')
        glb_path = generate_glb(
            predictions, target_dir, conf_thres, frame_filter,
            mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode
        )
        
        del predictions
        gc.collect()
        torch.cuda.empty_cache()
        
        update_progress(task_id, 'completed', 100, '3D reconstruction completed successfully,请在http://localhost:8080 处查看渲染结果', 
                       glb_path=glb_path, target_dir=target_dir)
        
        logger.info(f"3D reconstruction completed for task {task_id}")
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"3D reconstruction failed for task {task_id}: {error_trace}")
        update_progress(task_id, 'error', 0, f'Error: {str(e)}', error=str(e))


def start_async_reconstruction(task_id, video_path=None, image_paths=None, output_base_dir=None,
                               conf_thres=50.0, frame_filter="All", mask_black_bg=False, 
                               mask_white_bg=False, show_cam=True, mask_sky=False, 
                               prediction_mode="Depthmap and Camera Branch"):
    """
    Start asynchronous 3D reconstruction in a background thread.
    """
    thread = threading.Thread(
        target=process_3d_reconstruction,
        args=(task_id, video_path, image_paths, output_base_dir, 
              conf_thres, frame_filter, mask_black_bg, mask_white_bg, 
              show_cam, mask_sky, prediction_mode),
        daemon=True
    )
    thread.start()
    return thread


def update_glb_visualization(target_dir, conf_thres=50.0, frame_filter="All",
                            mask_black_bg=False, mask_white_bg=False, show_cam=True,
                            mask_sky=False, prediction_mode="Depthmap and Camera Branch"):
    """
    Update GLB visualization with new parameters without re-running inference.
    """
    if not target_dir or not os.path.isdir(target_dir):
        return None, "No valid target directory"
    
    predictions_path = os.path.join(target_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        return None, "No predictions found, please run reconstruction first"
    
    key_list = [
        "pose_enc", "depth", "depth_conf", "world_points", "world_points_conf",
        "images", "extrinsic", "intrinsic", "world_points_from_depth",
    ]
    
    loaded = np.load(predictions_path)
    predictions = {key: np.array(loaded[key]) for key in key_list}
    
    glb_path = generate_glb(
        predictions, target_dir, conf_thres, frame_filter,
        mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode
    )
    
    return glb_path, "Visualization updated"
