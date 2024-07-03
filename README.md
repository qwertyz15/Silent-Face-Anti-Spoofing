# Spoof Detection / Liveness Detection Model Inference

This repository contains scripts for testing and deploying models for spoof detection or liveness detection using PyTorch and ONNX.

## Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/qwertyz15/Silent-Face-Anti-Spoofing.git
   cd Silent-Face-Anti-Spoofing
   git checkout inference
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## PyTorch Models

### Testing

#### `test.py`

- Test a single image using the PyTorch model.
  ```bash
  python3 test.py --device_id <gpu_id> --model_dir <model_directory> --image_name <image_path>
  ```
  - `--device_id`: GPU ID to use (default: 0)
  - `--model_dir`: Path to the directory containing the PyTorch model (default: ./resources/anti_spoof_models)
  - `--image_name`: Path to the image to be tested (default: images/sample/image_T1.jpg)
  
#### `test_img_with_threshold.py`

- Test a single image with a custom liveness threshold.
  ```bash
  python3 test_img_with_threshold.py --device_id <gpu_id> --model_dir <model_directory> --image_name <image_path> --threshold <threshold_value>
  ```
  - `--device_id`: GPU ID to use (default: 0)
  - `--model_dir` or `-m`: Path to the directory containing the PyTorch model (default: ./resources/anti_spoof_models)
  - `--image_name` or `-i`: Path to the image to be tested (default: images/sample/image_F1.jpg)
  - `--threshold` or `-t`: Liveness threshold value (0 to 1) (default: 0.6)

#### `test_batch_infer.py`

- Perform batch testing on a folder containing images.
  ```bash
  python3 test_batch_infer.py --device_id <gpu_id> --model_dir <model_directory> --image_folder <image_folder_path> --batch_size <batch_size>
  ```
  - `--device_id`: GPU ID to use (default: 0)
  - `--model_dir`: Path to the directory containing the PyTorch model (default: ./resources/anti_spoof_models)
  - `--image_folder`: Path to the folder containing images for testing (required)
  - `--batch_size`: Batch size for testing (default: 1024)

#### `test_video_with_score.py`

- Test a video file for spoof detection with a custom liveness threshold.
  ```bash
  python3 test_video_with_score.py --device_id <gpu_id> --model_dir <model_directory> --video_path <video_path> --threshold <threshold_value>
  ```
  - `--device_id`: GPU ID to use (default: 0)
  - `--model_dir` or `-m`: Path to the directory containing the PyTorch model (default: ./resources/anti_spoof_models)
  - `--video_path` or `-v`: Path to the video file to be tested (default: 0)
  - `--threshold` or `-t`: Liveness threshold value (0 to 1) (default: 0.6)

#### `test_directory_batch_infer_voting.py`

- Perform batch testing on a folder containing images with a voting mechanism.
  ```bash
  python3 test_directory_batch_infer_voting.py --device_id <gpu_id> --model_dir <model_directory> --image_folder <image_folder_path> --batch_size <batch_size> --model_threshold <model_threshold_value> --voting_threshold <voting_threshold_value>
  ```
  - `--device_id`: GPU ID to use (default: 0)
  - `--model_dir`: Path to the directory containing the PyTorch model (default: ./resources/anti_spoof_models)
  - `--image_folder`: Path to the folder containing images for testing (required)
  - `--batch_size`: Batch size for testing (default: 1024)
  - `--model_threshold`: Liveness threshold for individual models (default: 0.75)
  - `--voting_threshold`: Threshold for considering the overall result as real (default: 0.8)

## ONNX Models

### Testing

#### `onnx_image_predict.py`

- Perform anti-spoofing detection on a single image using the ONNX model.
  ```bash
  python3 onnx_image_predict.py --threshold <threshold_value> --image_path <image_path> --model_dir <model_directory>
  ```
  - `--threshold`: Threshold for considering a prediction as real (default: 0.75)
  - `--image_path`: Path to the input image (default: ../images/sample/image_F2.jpg)
  - `--model_dir`: Directory containing ONNX models (default: ../resources/dynamic_onnx)

#### `onnx_video_predict.py`

- Perform anti-spoofing detection on a video stream using the ONNX model.
  ```bash
  python3 onnx_video_predict.py --threshold <threshold_value> --video_path <video_path> --model_dir <model_directory>
  ```
  - `--threshold`: Threshold for considering a prediction as real (default: 0.85)
  - `--video_path`: Path to the video file or camera index (default: 0)
  - `--model_dir`: Directory containing ONNX models (default: ../resources/dynamic_onnx)

#### `onnx_directory_predict.py`

- Process images and perform anti-spoofing detection using the ONNX model.
  ```bash
  python3 onnx_directory_predict.py --image_dir <image_folder_path> --model_dir <model_directory> --model_threshold <model_threshold_value> --voting_threshold <voting_threshold_value> --batch_size <batch_size>
  ```
  - `--image_dir`: Directory containing images (default: ../images/dataset_new/b)
  - `--model_dir`: Directory containing ONNX models (default: ../resources/dynamic_onnx)
  - `--model_threshold`: Threshold for considering a prediction as real (default: 0.75)
  - `--voting_threshold`: Threshold for considering the overall result as real (default: 0.8)
  - `--batch_size`: Batch size for processing images (default: 64)

## Flask Deployment

### PyTorch Model Deployment

#### `flask_app_pt.py`

- Deploy the PyTorch model as a Flask API for batch testing.
  ```bash
  python3 flask_app_pt.py --device_id <gpu_id> --model_dir <model_directory> --batch_size <batch_size> --model_threshold <model_threshold_value> --voting_threshold <voting_threshold_value>
  ```
  - `--device_id`: GPU device ID (default: 0)
  - `--model_dir`: Path to the model directory (default: ./resources/anti_spoof_models)
  - `--batch_size`: Batch size for testing (default: 64)
  - `--model_threshold`: Liveness threshold (default: 0.75)
  - `--voting_threshold`: Voting threshold (default: 0.8)

### ONNX Model Export
- Export the ONNX model
  ```bash
  python export.py --model_path path_to_model.pth --onnx_path output_model.onnx --device_id 0
  ```
### ONNX Model Deployment

#### `flask_app_onnx.py`

- Deploy the ONNX model as a Flask API for batch testing.
  ```bash
  python3 flask_app_onnx.py --model_dir <model_directory> --model_threshold <model_threshold_value> --voting_threshold <voting_threshold_value> --batch_size <batch_size> --device_id <gpu_id>
  ```
  - `--model_dir`: Directory containing ONNX models (default: ./resources/dynamic_onnx)
  - `--model_threshold`: Threshold for considering a prediction as real (default: 0.75)
  - `--voting_threshold`: Threshold for considering the overall result as real (default: 0.8)
  - `--batch_size`: Batch size for processing images (default: 64)
  - `--device_id`: Device ID for CUDA (default: 0)

## Flask Client

#### `flask_client_main.py`

- For testing both Flask applications (`flask_app_pt.py` and `flask_app_onnx.py`).
  ```bash
  python3 flask_client_main.py --image_dir <image_dir> --url <url>
  ```
  - `--image_dir`: Directory path containing the images to be processed.
  - `--url`: URL of the Flask server where the images will be sent for processing.
