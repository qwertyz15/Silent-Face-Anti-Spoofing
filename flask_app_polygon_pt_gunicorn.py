# gunicorn -w 4 -b 0.0.0.0:5000 flask_app_polygon_pt_gunicorn:app
import os
import cv2
import numpy as np
import base64
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

app = Flask(__name__)

model_test = None
image_cropper = None
all_models = {}
args = None

def init_app(device_id=0, model_dir="./resources/anti_spoof_models", batch_size=1024, model_threshold=0.8, voting_threshold=0.8):
    global model_test, image_cropper, all_models, args
    class Args:
        def __init__(self, device_id, model_dir, batch_size, model_threshold, voting_threshold):
            self.device_id = device_id
            self.model_dir = model_dir
            self.batch_size = batch_size
            self.model_threshold = model_threshold
            self.voting_threshold = voting_threshold

    args = Args(device_id, model_dir, batch_size, model_threshold, voting_threshold)

    model_test = AntiSpoofPredict(args.device_id)
    image_cropper = CropImage()

    # Load all models at startup
    for model_name in os.listdir(args.model_dir):
        model_path = os.path.join(args.model_dir, model_name)
        smodel = model_test._load_model(model_path)
        smodel.eval()
        all_models[model_name] = smodel

def decode_base64_img(base64_string):
    # Decode the Base64 string
    decoded_bytes = base64.b64decode(base64_string)
    # Convert bytes to numpy array
    nparr = np.frombuffer(decoded_bytes, np.uint8)
    # Decode numpy array to image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

def batch_predict(images, model_test, image_cropper, all_models, batch_size, threshold):
    results = []

    # Iterate over batches of images
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]  # Get the current batch of images
        batch_predictions = np.zeros((len(batch), 3))
        bboxes = model_test.get_bboxes(batch)  # Get bboxes for batch

        for model_name in all_models:
            smodel = all_models[model_name]
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            params_list = [{"org_img": img, "bbox": bbox, "scale": scale, "out_w": w_input, "out_h": h_input, "crop": True if scale is not None else False} for img, bbox in zip(batch, bboxes)]

            # Crop batch of images
            cropped_images = image_cropper.crop_batch(batch, params_list)

            # Predict batch for the current batch of images using the current model
            batch_predictions += model_test.predict_batch(cropped_images, smodel)

        # Determine results for each image in the batch
        for j, img in enumerate(batch):
            prediction = batch_predictions[j]
            label = np.argmax(prediction)
            value = prediction[label] / 2
            result = ("Real" if label == 1 else "Fake") + f" Score: {value:.2f}"
            results.append(result)

    return results

@app.route('/predict', methods=['POST'])
def predict():
    # The endpoint now uses the pre-loaded models without reloading them
    data = request.json
    base64_images = data['images']
    results = {}

    batch_size = args.batch_size
    threshold = args.model_threshold

    # Convert Base64 encoded images to OpenCV images
    images = [decode_base64_img(base64_img) for base64_img in base64_images]

    # Batch predict on converted images
    result_list = batch_predict(images, model_test, image_cropper, all_models, batch_size, threshold)

    total_image = len(result_list)
    total_real = sum(1 for result in result_list if "Real" in result)
    total_fake = sum(1 for result in result_list if "Fake" in result)
    
    liveness = total_real / total_image
    if liveness >= args.voting_threshold:
        prediction = {"prediction": "Real", "score": liveness}
    else:
        prediction = {"prediction": "Fake", "score": liveness}

    return jsonify(prediction)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Flask API for batch testing")
    parser.add_argument("--device_id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--model_dir", type=str, default="./resources/anti_spoof_models", help="Path to the model directory")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for testing")
    parser.add_argument("--model_threshold", type=float, default=0.8, help="Liveness threshold")
    parser.add_argument("--voting_threshold", type=float, default=0.8, help="Voting threshold")
    args = parser.parse_args()

    init_app(args.device_id, args.model_dir, args.batch_size, args.model_threshold, args.voting_threshold)
    
    app.run(host='0.0.0.0',debug=False)  # Run the Flask app
else:
    init_app()  # Initialize with default values when not running directly

