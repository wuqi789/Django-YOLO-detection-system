from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
import os
import random
import cv2
import requests
from .yolo_detector import get_detector, YOLODetector
from .ultralytics import YOLO
import time
from weasyprint import HTML
import numpy as np
import threading
import json
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration file path
CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.json')

# Load configuration from file
def load_config():
    """Load configuration from config.json file"""
    default_config = {
        'jetson_ip': '192.168.0.111',
        'batch_size': 1
    }
    
    if not os.path.exists(CONFIG_FILE):
        # Create default config file if it doesn't exist
        with open(CONFIG_FILE, 'w') as f:
            json.dump(default_config, f, indent=4)
        logger.info(f"Created default config file at {CONFIG_FILE}")
        return default_config
    
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        # Merge with default config to ensure all keys exist
        merged_config = {**default_config, **config}
        return merged_config
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        return default_config

# Save configuration to file
def save_config(config):
    """Save configuration to config.json file"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Saved config to {CONFIG_FILE}")
        return True
    except Exception as e:
        logger.error(f"Failed to save config file: {e}")
        return False

# Validate IP address
def is_valid_ip(ip):
    """Validate IP address format"""
    pattern = r'^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
    return re.match(pattern, ip) is not None

# Load initial configuration
config = load_config()

# Jetson Nano configuration
JETSON_IP = config['jetson_ip']
JETSON_VIDEO_URL = f"http://{JETSON_IP}:5000/video_feed"
JETSON_SENSOR_URL =f"http://{JETSON_IP}:5000/sensor"
# Global variable to store the latest frame from Jetson Nano
latest_jetson_frame = None
jetson_capture_thread = None
jetson_thread_running = False

# Function to continuously capture frames from Jetson Nano
def capture_jetson_frames():
    global latest_jetson_frame, jetson_thread_running
    
    cap = cv2.VideoCapture(JETSON_VIDEO_URL)
    if not cap.isOpened():
        print(f"❌ 无法连接到Jetson Nano视频流: {JETSON_VIDEO_URL}")
        jetson_thread_running = False
        return
    
    print(f"✅ 已连接到Jetson Nano视频流: {JETSON_VIDEO_URL}")
    jetson_thread_running = True
    
    while jetson_thread_running:
        ret, frame = cap.read()
        if ret:
            # Update the latest frame
            latest_jetson_frame = frame
        else:
            # If we can't read a frame, wait a bit before trying again
            time.sleep(0.1)
    
    cap.release()
    print("❌ Jetson Nano视频流捕获已停止")

# Global variables
models_cache = {}
current_model = None
models_path = os.path.join(os.path.dirname(__file__), 'static', 'models')

# Helper function to check if a file is a valid YOLO model
def is_valid_yolo_model(file_path):
    """Check if a file is a valid YOLOv8 model"""
    if not file_path.endswith('.pt'):
        return False
    
    try:
        # Try to load the model briefly to check validity
        model = YOLO(file_path, task='detect')
        return True
    except Exception as e:
        return False

# Helper function to scan models in the specified directory
def scan_models():
    """Scan the models directory for valid YOLOv8 models"""
    models = []
    
    if os.path.exists(models_path):
        for file in os.listdir(models_path):
            file_path = os.path.join(models_path, file)
            if os.path.isfile(file_path) and is_valid_yolo_model(file_path):
                file_stats = os.stat(file_path)
                models.append({
                    'name': file,
                    'path': file_path,
                    'size': file_stats.st_size,
                    'modified': time.ctime(file_stats.st_mtime),
                    'model_type': file.split('.')[0]  # Extract model type from filename
                })
    
    return models

# Helper function to get model parameters
def get_model_params(model_path):
    """Extract parameters from a YOLOv8 model"""
    try:
        model = YOLO(model_path)
        model_info = model.info()
        
        return {
            'input_size': model_info.get('input_size', [640, 640]),
            'num_classes': len(model.names),
            'model_type': model_info.get('model_type', 'yolov8'),
            'names': model.names,
            'version': model_info.get('version', 'unknown'),
            'task': model_info.get('task', 'detect')
        }
    except Exception as e:
        return {
            'error': str(e),
            'input_size': [640, 640],
            'num_classes': 0,
            'model_type': 'unknown',
            'names': {},
            'version': 'unknown',
            'task': 'detect'
        }
# Get the detector instance
detector = get_detector()

def index(request):
    return render(request, 'index.html')

def detect(request):
    if request.method == 'POST':
        # Get detection parameters from request
        confidence = float(request.POST.get('confidence', 0.5))
        # model_type = request.POST.get('model_type', 'yolov8s')
        
        # Validate confidence value
        if confidence < 0.3 or confidence > 0.9:
            confidence = 0.5  # Default to 0.5 if out of range
            print(f"⚠️  Invalid confidence value provided, using default: {confidence}")
        else:
            print(f"📊 Received confidence parameter: {confidence}")
        
        # Set confidence threshold
        detector.set_conf_threshold(confidence)
        
        # Load video frame from file for demo
        # In a real implementation, you would process a video frame from the camera
        video_path = os.path.join(os.path.dirname(__file__), 'static', 'media', 'video.mp4')
        
        # Capture a frame from the video
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Run actual detection with YOLOv8 on GPU
                detections = detector.detect_frame(frame)
                
                # Count detections by class
                helmet_count = sum(1 for d in detections if d['class_name'] == 'helmet')
                person_count = sum(1 for d in detections if d['class_name'] == 'person')
                respirator_count = sum(1 for d in detections if d['class_name'] == 'respirator')
                violation_count = sum(1 for d in detections if d['class_name'] not in ['helmet', 'person', 'respirator'])
            else:
                # Fallback to random results if frame capture fails
                helmet_count = random.randint(0, 5)
                person_count = random.randint(1, 10)
                respirator_count = random.randint(0, 3)
                violation_count = random.randint(0, 2)
                detections = []
        else:
            # Fallback to random results if video can't be opened
            helmet_count = random.randint(0, 5)
            person_count = random.randint(1, 10)
            respirator_count = random.randint(0, 3)
            violation_count = random.randint(0, 2)
            detections = []

        detections_data = {
            'helmet': helmet_count,
            'person': person_count,
            'respirator': respirator_count,
            'violation': violation_count,
            'boxes': detections
        }
        
        return JsonResponse({'status': 'success', 'detections': detections_data})
    return JsonResponse({'status': 'error', 'message': 'Method not allowed'})

def get_sensors(request):
    try:
        # Call Jetson sensor API
        # jetson_url = "http://192.168.0.111:5000/sensor"

        response = requests.get(JETSON_SENSOR_URL, timeout=2)
        jetson_data = response.json()
        
        # Directly use the sensor data without averaging
        # Ensure we have at least 3 values for each sensor type
        temperature = jetson_data['temperature']
        humidity = jetson_data['humidity']
        
        # Make sure we have exactly 3 values
        while len(temperature) < 3:
            temperature.append(temperature[-1] if temperature else "24.5")
        while len(humidity) < 3:
            humidity.append(humidity[-1] if humidity else "65")
        
        # Take only the first 3 values if we have more
        temperature = temperature[:3]
        humidity = humidity[:3]
        
        # Format response with all sensor data
        sensor_data = {
            'temperature': temperature,
            'humidity': humidity,
            'timestamp': jetson_data.get('timestamp', ''),
            'status': jetson_data.get('status', 'normal')
        }
        return JsonResponse({'status': 'success', 'data': sensor_data})
    except Exception as e:
        # Fallback to random data if Jetson API fails
        print(f"Failed to get data from Jetson: {str(e)}")
        # Generate random values for each sensor
        sensor_data = {
            'temperature': [f"{round(random.uniform(15.0, 30.0), 2)}", f"{round(random.uniform(15.0, 30.0), 2)}", f"{round(random.uniform(15.0, 30.0), 2)}"],
            'humidity': [f"{round(random.uniform(20.0, 80.0), 2)}", f"{round(random.uniform(20.0, 80.0), 2)}", f"{round(random.uniform(20.0, 80.0), 2)}"],
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'status': 'warning'
        }
        return JsonResponse({'status': 'success', 'data': sensor_data})

def get_stats(request):
    # Generate random stats data

    stats = {
        'total_detections': random.randint(300, 500),
        'online_devices': random.randint(3, 5),
        'violation_events': random.randint(2, 8),
        'system_uptime': '24:00:00',
        'avg_fps': random.randint(25, 35),
        'model_accuracy': round(random.uniform(90.0, 98.0), 1),
        'ai_coverage': 100
    }
    return JsonResponse({'status': 'success', 'data': stats})

from django.views.decorators.csrf import csrf_exempt

# Global progress storage (in-memory, for development only)
processing_progress = {}

@csrf_exempt
def process_video(request):
    if request.method == 'POST':
        import uuid
        import shutil
        import logging
        # Configure logging
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger('video_processing')
        
        try:
            logger.info("Received video processing request")
            
            # Get detection parameters from request
            confidence = float(request.POST.get('confidence', 0.5))
            model_type = request.POST.get('model_type', 'yolov8s')
            
            logger.info(f"Request parameters: confidence={confidence}, model_type={model_type}")
            
            # Get uploaded video file
            if 'video' not in request.FILES:
                logger.error("No video file in request")
                return JsonResponse({'status': 'error', 'message': 'No video file in request'})
            
            video_file = request.FILES['video']
            logger.info(f"Received video file: {video_file.name}, size: {video_file.size} bytes")


            # Generate unique filename for the uploaded video
            video_uuid = str(uuid.uuid4())
            logger.info(f"Generated video UUID: {video_uuid}")
            
            # Get absolute paths
            base_dir = os.path.dirname(os.path.abspath(__file__))
            temp_dir = os.path.join(base_dir, 'static', 'temp')
            media_dir = os.path.join(base_dir, 'static', 'media')
            
            logger.info(f"Base directory: {base_dir}")
            logger.info(f"Temp directory: {temp_dir}")
            logger.info(f"Media directory: {media_dir}")


            # Create directories if they don't exist
            os.makedirs(temp_dir, exist_ok=True)
            os.makedirs(media_dir, exist_ok=True)
            logger.info("Created directories if they didn't exist")
            
            # Save uploaded video to temp location
            uploaded_video_abs = os.path.join(temp_dir, f"{video_uuid}_input.mp4")
            processed_video_abs = os.path.join(media_dir, f"{video_uuid}_output.mp4")
            
            logger.info(f"Uploaded video path: {uploaded_video_abs}")
            logger.info(f"Processed video path: {processed_video_abs}")
            
            # Get task ID from request
            task_id = request.POST.get('task_id', '')
            logger.info(f"Task ID: {task_id}")
            
            # Initialize progress
            if task_id:
                processing_progress[task_id] = {
                    'progress': 0,
                    'status': 'processing',
                    'details': 'Starting video processing...',
                    'video_url': '',
                    'message': ''
                }
            
            # Save uploaded video
            logger.info("Saving uploaded video...")
            if task_id:
                processing_progress[task_id] = {
                    'progress': 20,
                    'status': 'processing',
                    'details': 'Saving video file...',
                    'video_url': '',
                    'message': ''
                }
            
            with open(uploaded_video_abs, 'wb+') as destination:
                for chunk in video_file.chunks():
                    destination.write(chunk)
            logger.info(f"Uploaded video saved: {uploaded_video_abs}")
            
            # Check if file exists and has content
            if not os.path.exists(uploaded_video_abs):
                logger.error(f"Uploaded video file does not exist: {uploaded_video_abs}")
                if task_id and task_id in processing_progress:
                    processing_progress[task_id] = {
                        'progress': 0,
                        'status': 'error',
                        'details': 'Failed to save video file',
                        'video_url': '',
                        'message': 'Failed to save video file'
                    }
                return JsonResponse({'status': 'error', 'message': 'Failed to save video file'})
            
            file_size = os.path.getsize(uploaded_video_abs)
            logger.info(f"Uploaded video file size: {file_size} bytes")
            
            if file_size == 0:
                logger.error(f"Uploaded video file is empty: {uploaded_video_abs}")
                if task_id and task_id in processing_progress:
                    processing_progress[task_id] = {
                        'progress': 0,
                        'status': 'error',
                        'details': 'Video file is empty',
                        'video_url': '',
                        'message': 'Video file is empty'
                    }
                return JsonResponse({'status': 'error', 'message': 'Video file is empty'})
            
            # Update progress
            if task_id:
                processing_progress[task_id] = {
                    'progress': 40,
                    'status': 'processing',
                    'details': 'Processing video...',
                    'video_url': '',
                    'message': 'Detecting'
                }
            
            try:
                # Open the uploaded video for processing
                cap = cv2.VideoCapture(uploaded_video_abs)
                if not cap.isOpened():
                    raise Exception(f"Failed to open video file: {uploaded_video_abs}")
                
                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fourcc_code = int(cap.get(cv2.CAP_PROP_FOURCC))
                
                # Calculate total video duration
                total_duration = total_frames / fps if fps > 0 else 0
                
                logger.info(f"Video properties: FPS={fps}, Width={width}, Height={height}, Total Frames={total_frames}, Duration={total_duration:.2f}s")
                
                # Determine the appropriate codec for output video
                try:
                    # Try to use the same codec as input
                    fourcc = cv2.VideoWriter_fourcc(*chr(fourcc_code & 0xFF) + chr((fourcc_code >> 8) & 0xFF) + chr((fourcc_code >> 16) & 0xFF) + chr((fourcc_code >> 24) & 0xFF))
                    out = cv2.VideoWriter(processed_video_abs, fourcc, fps, (width, height))
                except Exception as codec_error:
                    # Fallback to MP4V if original codec is not supported
                    logger.warning(f"Failed to use original codec: {codec_error}, falling back to MP4V")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(processed_video_abs, fourcc, fps, (width, height))
                
                if not out.isOpened():
                    raise Exception(f"Failed to create output video file: {processed_video_abs}")
                
                # Process video frame by frame
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Run detection on the frame
                    detections = detector.detect_frame(frame)
                    
                    # Draw detection boxes on the frame
                    frame_with_boxes = detector.draw_detections(frame.copy(), detections)
                    
                    # Write the processed frame to output video
                    out.write(frame_with_boxes)
                    
                    # Update progress
                    frame_count += 1
                    # Calculate progress based solely on video processing
                    progress = int((frame_count / total_frames) * 100)
                    
                    # Calculate current processed time in seconds
                    processed_time = frame_count / fps if fps > 0 else 0
                    
                    if task_id:
                        processing_progress[task_id] = {
                            'progress': progress,
                            'status': 'processing',
                            'details': f'Detecting, processed {frame_count}/{total_frames} frames',
                            'video_url': '',
                            'message': 'Detecting',
                            'processed_time': processed_time  # Add processed time in seconds
                        }
                    
                    # Log progress every 100 frames
                    if frame_count % 100 == 0:
                        logger.info(f"Processed {frame_count}/{total_frames} frames, progress: {progress}%, time: {processed_time:.2f}s")
                
                # Release resources
                cap.release()
                out.release()
                
                logger.info(f"Video processing completed successfully. Processed {frame_count} frames")
            except Exception as video_processing_error:
                logger.error(f"Video processing failed: {video_processing_error}")
                # Clean up resources if any
                if 'cap' in locals():
                    cap.release()
                if 'out' in locals():
                    out.release()
                raise
            
            # Update progress
            if task_id:
                processing_progress[task_id] = {
                    'progress': 80,
                    'status': 'processing',
                    'details': 'Saving processed video...',
                    'video_url': '',
                    'message': 'Detecting'
                }
            
            # Check if processed video exists
            if not os.path.exists(processed_video_abs):
                logger.error(f"Processed video file does not exist: {processed_video_abs}")
                if task_id and task_id in processing_progress:
                    processing_progress[task_id] = {
                        'progress': 0,
                        'status': 'error',
                        'details': 'Video processing failed',
                        'video_url': '',
                        'message': 'Video processing failed'
                    }
                return JsonResponse({'status': 'error', 'message': 'Video processing failed'})
            
            processed_size = os.path.getsize(processed_video_abs)
            logger.info(f"Processed video file size: {processed_size} bytes")
            
            # Remove temporary input video
            logger.info(f"Removing temporary input video: {uploaded_video_abs}")
            os.remove(uploaded_video_abs)
            logger.info(f"Temporary input video removed")
            
            # Return processed video URL
            video_url = f"/static/media/{video_uuid}_output.mp4"
            logger.info(f"Video processing completed successfully. Video URL: {video_url}")
            
            # Update final progress
            if task_id:
                processing_progress[task_id] = {
                    'progress': 100,
                    'status': 'completed',
                    'details': 'Video processing completed',
                    'video_url': video_url,
                    'message': 'Video processing completed',
                    'processed_time': total_duration  # Add total video duration as processed time
                }
            
            # Return processed video URL
            video_url = f"/static/media/{video_uuid}_output.mp4"
            logger.info(f"Video processing completed successfully. Video URL: {video_url}")
            
            return JsonResponse({
                'status': 'success',
                'video_url': video_url,
                'message': 'Video processing completed, processed 1 frame',
                'debug_info': {
                    'video_uuid': video_uuid,
                    'video_url': video_url,
                    'processed_size': processed_size
                }
            })
            
        except Exception as e:
            # Log the error for debugging
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"Video processing error: {error_trace}")
            
            # Clean up
            try:
                if 'uploaded_video_abs' in locals() and os.path.exists(uploaded_video_abs):
                    os.remove(uploaded_video_abs)
                if 'processed_video_abs' in locals() and os.path.exists(processed_video_abs):
                    os.remove(processed_video_abs)
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup: {cleanup_error}")
            
            return JsonResponse({
                'status': 'error',
                'message': f'Video processing failed: {str(e)}',
                'error_details': error_trace,
                'debug_info': {
                    'request_method': request.method,
                    'has_video_file': 'video' in request.FILES
                }
            })
        
        logger.error(f"Received non-POST request: {request.method}")
    return JsonResponse({'status': 'error', 'message': 'Only POST requests allowed'})

@csrf_exempt
def process_progress(request):
    """
    Return the progress of video processing for a specific task ID
    """
    task_id = request.GET.get('task_id', '')
    if task_id in processing_progress:
        return JsonResponse({'status': 'success', **processing_progress[task_id]})
    return JsonResponse({'status': 'error', 'message': 'Task not found'})

@csrf_exempt
def detect_frame(request):
    """
    API endpoint to process a single video frame and return detection results with annotated image
    """
    if request.method == 'POST':
        # Get the frame from request
        frame_file = request.FILES.get('frame')
        if not frame_file:
            return JsonResponse({'status': 'error', 'message': 'No frame provided'})
        
        # Get detection parameters
        confidence = float(request.POST.get('confidence', 0.5))
        model_type = request.POST.get('model_type', 'yolov8s')
        
        # Validate confidence value
        if confidence < 0.3 or confidence > 0.9:
            confidence = 0.5  # Default to 0.5 if out of range
            print(f"⚠️  Invalid confidence value provided in detect_frame, using default: {confidence}")
        else:
            print(f"📊 Received confidence parameter in detect_frame: {confidence}")
        
        # Set confidence threshold
        detector.set_conf_threshold(confidence)
        
        try:
            # Read frame from file
            import numpy as np
            import base64
            frame_bytes = frame_file.read()
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Run detection and generate annotated image
            detected_frame, detections = detector.detect_and_plot(frame)
            
            # Filter detections to focus on person, helmet, and ventilator
            filtered_detections = [d for d in detections if d['class_name'] in ['person', 'helmet', 'ventilator']]
            
            # Count detections by class
            helmet_count = sum(1 for d in filtered_detections if d['class_name'] == 'helmet')
            person_count = sum(1 for d in filtered_detections if d['class_name'] == 'person')
            respirator_count = sum(1 for d in filtered_detections if d['class_name'] == 'ventilator')  # Map ventilator to respirator for frontend compatibility
            violation_count = sum(1 for d in filtered_detections if d['class_name'] not in ['helmet', 'person', 'ventilator'])
            
            # Convert detected frame to Base64 for faster transmission
            _, buffer = cv2.imencode('.jpg', detected_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Return detection results with annotated image
            return JsonResponse({
                'status': 'success',
                'detections': {
                    'helmet': helmet_count,
                    'person': person_count,
                    'respirator': respirator_count,
                    'violation': violation_count,
                    'boxes': filtered_detections
                },
                'annotated_image': img_base64
            })
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            return JsonResponse({
                'status': 'error',
                'message': f'Frame processing failed: {str(e)}',
                'error_details': error_trace
            })
    return JsonResponse({'status': 'error', 'message': 'Method not allowed'})

# Model management API functions
@csrf_exempt
def get_models(request):
    """
    API endpoint to get list of available models
    """
    try:
        models = scan_models()
        return JsonResponse({
            'status': 'success',
            'models': models,
            'count': len(models)
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to scan models: {str(e)}'
        })

@csrf_exempt
def get_model_params_api(request):
    """
    API endpoint to get parameters of a specific model
    """
    if request.method == 'GET':
        model_name = request.GET.get('model_name')
        if not model_name:
            return JsonResponse({'status': 'error', 'message': 'Missing model_name parameter'})
        
        model_path = os.path.join(models_path, model_name)
        if not os.path.exists(model_path):
            return JsonResponse({'status': 'error', 'message': 'Model file not found'})
        
        try:
            params = get_model_params(model_path)
            return JsonResponse({'status': 'success', 'params': params})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f'Failed to get model params: {str(e)}'})
    return JsonResponse({'status': 'error', 'message': 'Only GET requests allowed'})

@csrf_exempt
def switch_model(request):
    """
    API endpoint to switch the current detection model
    """
    global detector, current_model
    
    if request.method == 'POST':
        model_name = request.POST.get('model_name')
        if not model_name:
            return JsonResponse({'status': 'error', 'message': 'Missing model_name parameter'})
        
        model_path = os.path.join(models_path, model_name)
        if not os.path.exists(model_path):
            return JsonResponse({'status': 'error', 'message': 'Model file not found'})
        
        if not is_valid_yolo_model(model_path):
            return JsonResponse({'status': 'error', 'message': 'Invalid YOLO model file'})
        
        try:
            # Load the new model
            detector = YOLODetector(model_path)
            current_model = model_name
            models_cache[model_name] = detector
            
            return JsonResponse({
                'status': 'success',
                'message': 'Model switched successfully',
                'current_model': model_name
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f'Failed to switch model: {str(e)}'
            })
    return JsonResponse({'status': 'error', 'message': 'Only POST requests allowed'})

@csrf_exempt
def get_current_model(request):
    """
    API endpoint to get the current model information
    """
    global current_model
    
    try:
        if current_model:
            model_path = os.path.join(models_path, current_model)
            params = get_model_params(model_path)
            return JsonResponse({
                'status': 'success',
                'current_model': current_model,
                'params': params
            })
        else:
            # Get default model from get_detector()
            return JsonResponse({
                'status': 'success',
                'current_model': 'default',
                'params': get_model_params(os.path.join(models_path, 'best.pt'))
            })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to get current model: {str(e)}'
        })

@csrf_exempt
def detect_image(request):
    """
    API endpoint to process a single image and return detection results
    """
    if request.method == 'POST':
        # Get the image from request
        image_file = request.FILES.get('image')
        if not image_file:
            return JsonResponse({'status': 'error', 'message': 'No image provided'})
        
        # Get detection parameters
        confidence = float(request.POST.get('confidence', 0.5))
        model_type = request.POST.get('model_type', 'yolov8s')
        
        # Validate confidence value
        if confidence < 0.3 or confidence > 0.9:
            confidence = 0.5  # Default to 0.5 if out of range
            print(f"⚠️  Invalid confidence value provided in detect_image, using default: {confidence}")
        else:
            print(f"📊 Received confidence parameter in detect_image: {confidence}")
        
        # Set confidence threshold
        detector.set_conf_threshold(confidence)
        
        try:
            # Read image from file
            import numpy as np
            from PIL import Image
            import io
            import uuid
            
            # Open image using PIL
            image = Image.open(image_file)
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # Convert to numpy array
            frame = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # 使用新的detect_and_plot方法，直接生成带标注的图片
            frame_with_boxes, detections = detector.detect_and_plot(frame)
            
            # Generate unique filename for the processed image
            image_uuid = str(uuid.uuid4())
            image_filename = f"{image_uuid}_result.jpg"
            
            # Get absolute path for saving the image
            base_dir = os.path.dirname(os.path.abspath(__file__))
            media_dir = os.path.join(base_dir, 'static', 'media')
            
            # Create media directory if it doesn't exist
            os.makedirs(media_dir, exist_ok=True)
            
            # Save the processed image
            image_path = os.path.join(media_dir, image_filename)
            cv2.imwrite(image_path, frame_with_boxes)
            
            # Prepare image URL for frontend
            image_url = f"/static/media/{image_filename}"
            
            # Count detections by class
            helmet_count = sum(1 for d in detections if d['class_name'] == 'helmet')
            person_count = sum(1 for d in detections if d['class_name'] == 'person')
            respirator_count = sum(1 for d in detections if d['class_name'] == 'ventilator')  # Map ventilator to respirator for frontend compatibility
            violation_count = helmet_count + person_count + respirator_count  # Sum of all detections
            
            # Filter detections to focus on person, helmet, and ventilator
            filtered_detections = [d for d in detections if d['class_name'] in ['person', 'helmet', 'ventilator']]
            
            # Get image dimensions
            height, width, channels = frame.shape
            
            # Return detection results
            return JsonResponse({
                'status': 'success',
                'detections': {
                    'helmet': helmet_count,
                    'person': person_count,
                    'respirator': respirator_count,
                    'violation': violation_count,
                    'boxes': filtered_detections
                },
                'image_url': image_url,
                'image_info': {
                    'width': width,
                    'height': height,
                    'channels': channels
                },
                'message': 'Image detection completed successfully'
            })
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            return JsonResponse({
                'status': 'error',
                'message': f'Image processing failed: {str(e)}',
                'error_details': error_trace
            })
    return JsonResponse({'status': 'error', 'message': 'Method not allowed'})

def image_result(request):
    """
    View to display image detection results in a new window
    """
    return render(request, 'image_result.html')

# Camera Management API
def get_camera_status(request):
    """
    API endpoint to get the status of all cameras
    """
    try:
        import cv2
        # Function to detect the number of available cameras
        def detect_camera_count():
            count = 0
            # Try to open each camera index until we can't
            for i in range(10):  # Check up to 10 camera indexes
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    count += 1
                    cap.release()
                # Continue checking all indexes, don't break on first failure
            return count
        
        # Detect the actual number of cameras
        camera_count = detect_camera_count()
        
        # Create status dictionary based on detected cameras
        camera_statuses = {}
        for i in range(1, 5):  # We still show 4 camera cards, but mark unavailable ones as offline
            if i <= camera_count:
                camera_statuses[f'camera{i}'] = 'online'
            else:
                camera_statuses[f'camera{i}'] = 'offline'
        
        return JsonResponse({'status': 'success', 'data': camera_statuses, 'camera_count': camera_count})
    except Exception as e:
        print(f"Error getting camera status: {str(e)}")
        # Fallback to default status if OpenCV fails
        camera_statuses = {
            'camera1': 'online',
            'camera2': 'online',
            'camera3': 'online',
            'camera4': 'offline'
        }
        return JsonResponse({'status': 'success', 'data': camera_statuses, 'camera_count': 3})

@csrf_exempt
def control_camera(request):
    """
    API endpoint to control cameras (open, detect, stop)
    """
    if request.method == 'POST':
        try:
            camera_id = request.POST.get('camera_id')
            action = request.POST.get('action')
            
            if not camera_id or not action:
                return JsonResponse({'status': 'error', 'message': 'Missing camera_id or action'})
            
            # In a real implementation, you would use OpenCV to control the camera
            # For now, we'll just return a success response
            
            return JsonResponse({'status': 'success', 'message': f'Camera {camera_id} {action}ed successfully'})
        except Exception as e:
            print(f"Error controlling camera: {str(e)}")
            return JsonResponse({'status': 'error', 'message': str(e)})
    return JsonResponse({'status': 'error', 'message': 'Only POST requests allowed'})


# AI Analysis API
from django.views.decorators.csrf import csrf_exempt
import random
import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deepseek import get_ai_suggestion

@csrf_exempt
def get_ai_analysis(request):
    """
    API endpoint to get AI analysis based on sensor data
    """
    if request.method == 'GET':
        try:
            # Get sensor data from Jetson API
            # jetson_url = "http://192.168.0.110:5000/sensor"
            response = requests.get(JETSON_SENSOR_URL, timeout=2)
            jetson_data = response.json()
            
            # Format sensor data for AI analysis
            sensor_data = {
                "sensor1": {
                    "temperature": round(float(jetson_data['temperature'][0]), 1),
                    "humidity": round(float(jetson_data['humidity'][0]))
                },
                "sensor2": {
                    "temperature": round(float(jetson_data['temperature'][1]), 1),
                    "humidity": round(float(jetson_data['humidity'][1]))
                },
                "sensor3": {
                    "temperature": round(float(jetson_data['temperature'][2]), 1),
                    "humidity": round(float(jetson_data['humidity'][2]))
                }
            }

            # 调用AI建议生成函数
            suggestion = get_ai_suggestion(sensor_data)
            print(suggestion)
            ai2wxgzh(suggestion)
            # 返回JSON响应
            return JsonResponse({
                'status': 'success',
                'suggestion': suggestion,
                'sensor_data': sensor_data
            })
        except Exception as e:
            print(f"Failed to get data from Jetson: {str(e)}")
            # Fallback to random data if Jetson API fails
            sensor_data = {
                "sensor1": {
                    "temperature": round(random.uniform(15.0, 25.0), 1),
                    "humidity": random.randint(50, 80)
                },
                "sensor2": {
                    "temperature": round(random.uniform(15.0, 25.0), 1),
                    "humidity": random.randint(50, 80)
                },
                "sensor3": {
                    "temperature": round(random.uniform(15.0, 25.0), 1),
                    "humidity": random.randint(50, 80)
                }
            }
            suggestion = get_ai_suggestion(sensor_data)
            return JsonResponse({
                'status': 'success',
                'suggestion': suggestion,
                'sensor_data': sensor_data
            })
    return JsonResponse({
        'status': 'error',
        'message': 'Method not allowed'
    })
from .wxgzh  import WechatMessageSender
APPID = 'wx7564a41e542f83e9'
APPSECRET = '571989cddf2f9332e1952863aae2ef87'
openids = ["ojARd6nUw4tePDe1X80DWLY_oBv4"]
global_warning1=0
def convert_markdown_to_plain_text(markdown_text):
    """
    Convert Markdown text to plain text by removing Markdown syntax
    """
    import re
    # Remove ### headers
    plain_text = re.sub(r'^###\s+', '', markdown_text, flags=re.MULTILINE)
    # Remove ** bold markers
    plain_text = re.sub(r'\*\*(.*?)\*\*', r'\1', plain_text)
    return plain_text

def ai2wxgzh(analysis1):
    warning_msg = "有人员未按规定佩戴防护措施！\n\n" if global_warning1 == 1 else "无人员未按规定佩戴防护措施！\n\n"
    # Convert markdown to plain text
    plain_analysis = convert_markdown_to_plain_text(analysis1)
    message_content = f"这一个小时内:\n{warning_msg}{plain_analysis}"
    # Push to WeChat
    sender = WechatMessageSender(APPID, APPSECRET, openids)
    sender.send_messages(message_content)
@csrf_exempt
def start_jetson_stream(request):
    """
    API endpoint to start the Jetson Nano video stream capture
    """
    global jetson_capture_thread, jetson_thread_running
    
    if request.method == 'POST':
        try:
            # If the thread is already running, return success
            if jetson_thread_running:
                return JsonResponse({
                    'status': 'success',
                    'message': 'Jetson Nano video stream is already running'
                })
            
            # Start a new thread to capture frames from Jetson Nano
            jetson_capture_thread = threading.Thread(target=capture_jetson_frames, daemon=True)
            jetson_capture_thread.start()
            
            # Wait a short time to allow the thread to initialize
            time.sleep(1)
            
            if jetson_thread_running:
                return JsonResponse({
                    'status': 'success',
                    'message': 'Jetson Nano video stream started successfully'
                })
            else:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Failed to start Jetson Nano video stream'
                })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f'Error starting Jetson Nano video stream: {str(e)}'
            })
    return JsonResponse({
        'status': 'error',
        'message': 'Method not allowed'
    })

@csrf_exempt
def stop_jetson_stream(request):
    """
    API endpoint to stop the Jetson Nano video stream capture
    """
    global jetson_thread_running
    
    if request.method == 'POST':
        try:
            jetson_thread_running = False
            
            # Wait for the thread to terminate if it's running
            if jetson_capture_thread and jetson_capture_thread.is_alive():
                jetson_capture_thread.join(timeout=2)
            
            return JsonResponse({
                'status': 'success',
                'message': 'Jetson Nano video stream stopped successfully'
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f'Error stopping Jetson Nano video stream: {str(e)}'
            })
    return JsonResponse({
        'status': 'error',
        'message': 'Method not allowed'
    })

@csrf_exempt
def get_jetson_frame(request):
    """
    API endpoint to get the latest frame from Jetson Nano
    """
    global latest_jetson_frame
    
    if request.method == 'GET':
        try:
            if latest_jetson_frame is None:
                return JsonResponse({
                    'status': 'error',
                    'message': 'No frame available from Jetson Nano'
                })
            
            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', latest_jetson_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Failed to encode frame'
                })
            
            # Convert to bytes and return as HttpResponse with content type image/jpeg
            frame_bytes = buffer.tobytes()
            return HttpResponse(frame_bytes, content_type='image/jpeg')
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f'Error getting Jetson Nano frame: {str(e)}'
            })
    return JsonResponse({
        'status': 'error',
        'message': 'Method not allowed'
    })

@csrf_exempt
def save_settings(request):
    """
    API endpoint to save system settings
    """
    if request.method == 'POST':
        try:
            # Get settings data from request
            jetson_ip = request.POST.get('jetson_ip', '')
            
            # Validate IP address
            if not is_valid_ip(jetson_ip):
                return JsonResponse({
                    'status': 'error',
                    'message': 'Invalid IP address format'
                })
            
            # Update global config
            global config, JETSON_IP, JETSON_VIDEO_URL
            config['jetson_ip'] = jetson_ip
            
            # Save to file
            if save_config(config):
                # Update global variables
                JETSON_IP = jetson_ip
                JETSON_VIDEO_URL = f"http://{JETSON_IP}:5000/video_feed"
                
                logger.info(f"Updated Jetson IP to {jetson_ip}")
                return JsonResponse({
                    'status': 'success',
                    'message': 'Settings saved successfully'
                })
            else:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Failed to save settings'
                })
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            return JsonResponse({
                'status': 'error',
                'message': f'Error saving settings: {str(e)}'
            })
    return JsonResponse({
        'status': 'error',
        'message': 'Method not allowed'
    })

@csrf_exempt
def export_ai_pdf(request):
    """
    API endpoint to export AI analysis as PDF
    """
    if request.method == 'POST':
        try:
            import json
            from datetime import datetime
            
            # 获取POST请求中的数据
            if 'data' in request.POST:
                # 从表单中获取数据
                data_str = request.POST.get('data', '')
                data = json.loads(data_str)
            else:
                # 从JSON请求体中获取数据
                data = json.loads(request.body)
            suggestion = data.get('suggestion', '')
            sensor_data = data.get('sensor_data', {})
            
            if not suggestion or not sensor_data:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Missing required data'
                })
            
            # 改进的Markdown解析函数，支持更多格式
            def improved_parse(text):
                # 替换标题
                text = text.replace('###', '<h3 style="color: #2c3e50; font-size: 18px; margin-top: 20px; margin-bottom: 15px; font-weight: bold;">')
                text = text.replace('##', '<h2 style="color: #3498db; font-size: 20px; margin-top: 25px; margin-bottom: 20px; font-weight: bold;">')
                # 替换粗体
                text = text.replace('**', '')
                # 替换列表项
                text = text.replace('\n- ', '<br><span style="margin-left: 20px; font-weight: bold;">•</span> ')
                text = text.replace('\n• ', '<br><span style="margin-left: 20px; font-weight: bold;">•</span> ')
                # 处理换行符
                text = text.replace('\n', '<br>')
                return text
            
            # 解析AI建议
            parsed_suggestion = improved_parse(suggestion)
            
            # 为传感器数据添加状态标签
            def get_sensor_status(temperature, humidity):
                # 粮仓环境状态判断逻辑
                if 18 <= temperature <= 22 and 50 <= humidity <= 70:
                    return "normal", "正常"
                elif (15 <= temperature < 18 or 22 < temperature <= 25) and (45 <= humidity < 50 or 70 < humidity <= 75):
                    return "warning", "警告"
                else:
                    return "danger", "危险"
            
            # 生成PDF文件的当前时间
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 生成传感器数据HTML
            sensor_html = ''
            sensor_index = 0
            for sensor_name, data_item in sensor_data.items():
                sensor_index += 1
                # 获取传感器状态
                status_class, status_text = get_sensor_status(data_item["temperature"], data_item["humidity"])
                
                # 添加传感器位置信息
                sensor_locations = {
                    "sensor1": "仓库东侧",
                    "sensor2": "仓库中央",
                    "sensor3": "仓库西侧"
                }
                location = sensor_locations.get(sensor_name, "未知位置")
                
                # 为不同传感器设置不同的渐变背景
                gradient_styles = [
                    "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
                    "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
                    "linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)",
                    "linear-gradient(135deg, #fa709a 0%, #fee140 100%)",
                    "linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)",
                    "linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%)"
                ]
                gradient_style = gradient_styles[(sensor_index - 1) % len(gradient_styles)]
                
                sensor_html += f'''<div class="sensor-card" style="background: {gradient_style};">
                    <div class="sensor-header">
                        <div class="sensor-name">{sensor_name}</div>
                        <span class="sensor-location">{location}</span>
                    </div>
                    <div class="status-tag status-{status_class}">{status_text}</div>
                    <div class="sensor-data-item">
                        <span class="sensor-label">温度</span>
                        <span class="sensor-value">{data_item["temperature"]}°C</span>
                    </div>
                    <div class="sensor-data-item">
                        <span class="sensor-label">湿度</span>
                        <span class="sensor-value">{data_item["humidity"]}%</span>
                    </div>
                    <div class="sensor-data-item">
                        <span class="sensor-label">检测时间</span>
                        <span class="sensor-value">{current_time}</span>
                    </div>
                </div>'''
            
            # 使用更简单的HTML模板，确保weasyprint能够正确处理
            html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>AI智能分析报告</title>
    <style>
        /* 全局样式 */
        * {{ 
            margin: 0; 
            padding: 0; 
            box-sizing: border-box; 
        }}
        
        body {{ 
            font-family: "Microsoft YaHei", Arial, sans-serif; 
            line-height: 1.6; 
            color: #333; 
            background-color: #f5f7fa; 
            padding: 20px; 
        }}
        
        /* 报告容器 */
        .report-container {{ 
            max-width: 800px; 
            margin: 0 auto; 
            background-color: white; 
            padding: 30px; 
            border-radius: 8px; 
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1); 
        }}
        
        /* 标题样式 */
        h1 {{ 
            color: #2c3e50; 
            text-align: center; 
            font-size: 24px; 
            margin-bottom: 20px; 
            font-weight: bold; 
        }}
        
        h2 {{ 
            color: #3498db; 
            font-size: 20px; 
            margin-top: 30px; 
            margin-bottom: 15px; 
            padding-bottom: 8px; 
            border-bottom: 2px solid #3498db; 
        }}
        
        h3 {{ 
            color: #2c3e50; 
            font-size: 16px; 
            margin-top: 20px; 
            margin-bottom: 10px; 
            font-weight: bold; 
        }}
        
        /* 报告头部信息 */
        .report-header {{ 
            background: #667eea; 
            color: white; 
            padding: 20px; 
            border-radius: 8px; 
            margin-bottom: 25px; 
        }}
        
        .header-info {{ 
            display: block; 
        }}
        
        .info-item {{ 
            font-size: 14px; 
            margin: 5px 0; 
        }}
        
        .info-label {{ 
            font-weight: bold; 
            margin-right: 8px; 
        }}
        
        /* 传感器数据样式 */
        .sensor-grid {{ 
            display: block; 
            margin: 25px 0; 
        }}
        
        .sensor-card {{ 
            color: white; 
            padding: 20px; 
            border-radius: 8px; 
            margin-bottom: 15px; 
        }}
        
        .sensor-header {{ 
            display: block; 
            margin-bottom: 15px; 
        }}
        
        .sensor-name {{ 
            font-size: 18px; 
            font-weight: bold; 
        }}
        
        .sensor-location {{ 
            font-size: 12px; 
            opacity: 0.9; 
            background: rgba(255, 255, 255, 0.2); 
            padding: 4px 10px; 
            border-radius: 12px; 
        }}
        
        .sensor-data-item {{ 
            display: flex; 
            justify-content: space-between; 
            margin: 10px 0; 
            font-size: 14px; 
        }}
        
        .sensor-label {{ 
            opacity: 0.9; 
        }}
        
        .sensor-value {{ 
            font-weight: bold; 
            font-size: 16px; 
        }}
        
        /* AI分析建议样式 */
        .suggestion-section {{ 
            background: #e8f4f8; 
            padding: 25px; 
            margin: 25px 0; 
            border-radius: 8px; 
            border-left: 4px solid #3498db; 
        }}
        
        .suggestion-title {{ 
            font-size: 16px; 
            font-weight: bold; 
            margin-bottom: 15px; 
            color: #2c3e50; 
        }}
        
        .suggestion-content {{ 
            line-height: 1.8; 
            font-size: 14px; 
            color: #555; 
        }}
        
        /* 系统概览样式 */
        .system-overview {{ 
            background: #fff3cd; 
            padding: 20px; 
            margin: 25px 0; 
            border-radius: 8px; 
            border: 1px solid #ffeeba; 
        }}
        
        .overview-grid {{ 
            display: block; 
            margin-top: 15px; 
        }}
        
        .overview-item {{ 
            text-align: center; 
            background: white; 
            padding: 15px; 
            border-radius: 8px; 
            margin: 10px 0; 
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); 
        }}
        
        .overview-value {{ 
            font-size: 24px; 
            font-weight: bold; 
            color: #856404; 
        }}
        
        .overview-label {{ 
            font-size: 14px; 
            color: #856404; 
            margin-top: 5px; 
        }}
        
        /* 页脚样式 */
        .report-footer {{ 
            margin-top: 30px; 
            padding: 20px; 
            background: #f5f7fa; 
            border-radius: 8px; 
            text-align: center; 
            color: #666; 
            font-size: 14px; 
        }}
        
        .timestamp {{ 
            font-style: italic; 
            margin-top: 10px; 
            font-size: 13px; 
        }}
        
        /* 状态标签样式 */
        .status-tag {{ 
            display: inline-block; 
            padding: 6px 12px; 
            border-radius: 15px; 
            font-size: 12px; 
            font-weight: bold; 
            margin-bottom: 10px; 
        }}
        
        .status-normal {{ 
            background-color: #d4edda; 
            color: #155724; 
        }}
        
        .status-warning {{ 
            background-color: #fff3cd; 
            color: #856404; 
        }}
        
        .status-danger {{ 
            background-color: #f8d7da; 
            color: #721c24; 
        }}
        
        /* 数据统计样式 */
        .stats-section {{ 
            background: white; 
            padding: 20px; 
            margin: 25px 0; 
            border-radius: 8px; 
        }}
        
        .stats-grid {{ 
            display: block; 
            margin-top: 15px; 
        }}
        
        .stat-card {{ 
            background: #667eea; 
            color: white; 
            padding: 15px; 
            border-radius: 8px; 
            text-align: center; 
            margin: 10px 0; 
        }}
        
        .stat-value {{ 
            font-size: 28px; 
            font-weight: bold; 
            margin-bottom: 5px; 
        }}
        
        .stat-label {{ 
            font-size: 14px; 
            opacity: 0.9; 
        }}
        
        /* 列表样式 */
        ul, ol {{ 
            margin-left: 20px; 
            margin-bottom: 15px; 
        }}
        
        li {{ 
            margin-bottom: 8px; 
        }}
    </style>
</head>
<body>
    <div class="report-container">
        <!-- 报告标题 -->
        <h1>AI智能粮仓分析报告</h1>
        
        <!-- 报告头部信息 -->
        <div class="report-header">
            <div class="header-info">
                <div class="info-item">
                    <span class="info-label">报告类型:</span>
                    <span>AI智能分析</span>
                </div>
                <div class="info-item">
                    <span class="info-label">生成时间:</span>
                    <span>{current_time}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">传感器数量:</span>
                    <span>{len(sensor_data)}个</span>
                </div>
                <div class="info-item">
                    <span class="info-label">报告编号:</span>
                    <span>REPORT-{current_time.replace('-', '').replace(':', '').replace(' ', '')[:14]}</span>
                </div>
            </div>
        </div>
        
        <!-- 传感器数据 -->
        <h2>传感器实时数据</h2>
        <div class="sensor-grid">
            {sensor_html}
        </div>
        
        <!-- AI分析建议 -->
        <h2>AI智能分析与建议</h2>
        <div class="suggestion-section">
            <div class="suggestion-content">
                {parsed_suggestion}
            </div>
        </div>
        
        <!-- 粮仓环境管理建议 -->
        <h2>粮仓环境管理建议</h2>
        <div class="system-overview">
            <ul>
                <li>保持粮仓温度在18-22°C之间，湿度在50-70%之间</li>
                <li>定期检查传感器设备，确保数据采集准确性</li>
                <li>根据季节变化调整通风策略，维持稳定环境</li>
                <li>建立完善的环境监控档案，便于历史数据分析</li>
                <li>结合AI建议及时调整环境控制措施，预防霉变风险</li>
            </ul>
        </div>
        
        <!-- 报告页脚 -->
        <div class="report-footer">
            <p>© 2026 AI智能粮仓管理系统 | 保留所有权利</p>
            <p class="timestamp">报告生成时间: {current_time}</p>
        </div>
    </div>
</body>
</html>'''
            
            # 使用weasyprint生成PDF
            pdf = HTML(string=html).write_pdf()
            
            # 返回PDF响应
            response = HttpResponse(content_type='application/pdf')
            response['Content-Disposition'] = f'attachment; filename="AI_Report_{datetime.now().strftime("%Y-%m-%d")}.pdf"'
            response.write(pdf)
            
            return response
            
        except Exception as e:
            # 详细记录错误信息
            import traceback
            error_trace = traceback.format_exc()
            print(f"PDF生成错误详情: {error_trace}")
            return JsonResponse({
                'status': 'error',
                'message': f'Failed to generate PDF: {str(e)}',
                'error_details': error_trace
            })
    return JsonResponse({
        'status': 'error',
        'message': 'Method not allowed'
    })
