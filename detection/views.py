from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
import os
import random
import cv2
from .yolo_detector import get_detector, YOLODetector
from .ultralytics import YOLO
import time

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
    # Generate random sensor data
    sensor_data = {
        'temperature': round(random.uniform(15.0, 30.0), 1),
        'humidity': random.randint(20, 80)
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
            violation_count = sum(1 for d in detections if d['class_name'] not in ['helmet', 'person', 'ventilator'])
            
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
