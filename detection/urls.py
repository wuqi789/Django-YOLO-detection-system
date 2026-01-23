from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('detect/', views.detect, name='detect'),
    path('detect_frame/', views.detect_frame, name='detect_frame'),
    path('detect_image/', views.detect_image, name='detect_image'),
    path('image_result/', views.image_result, name='image_result'),
    path('get_sensors/', views.get_sensors, name='get_sensors'),
    path('get_stats/', views.get_stats, name='get_stats'),
    path('process_video/', views.process_video, name='process_video'),
    path('process_progress/', views.process_progress, name='process_progress'),
    # Camera management API endpoints
    # path('get_camera_status/', views.get_camera_status, name='get_camera_status'),
    path('control_camera/', views.control_camera, name='control_camera'),
    # Model management API endpoints
    path('get_models/', views.get_models, name='get_models'),
    path('get_model_params/', views.get_model_params_api, name='get_model_params'),
    path('switch_model/', views.switch_model, name='switch_model'),
    path('get_current_model/', views.get_current_model, name='get_current_model'),
    # AI Analysis API endpoint
    path('get_ai_analysis/', views.get_ai_analysis, name='get_ai_analysis'),
    path('export_ai_pdf/', views.export_ai_pdf, name='export_ai_pdf'),
] 