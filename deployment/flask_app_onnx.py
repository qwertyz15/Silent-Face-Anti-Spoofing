import argparse
import os
import cv2
import sys
import numpy as np
import torch
import onnxruntime
from PIL import Image
import base64
from io import BytesIO
from flask import Flask, request, jsonify
from src.generate_patches import CropImage
from src.utility import parse_model_name
from src.anti_spoof_predict import AntiSpoofPredict

app = Flask(__name__)

def load_model_info(model_path):
    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return {
        "session": session,
        "input_name": input_name,
        "output_name": output_name
    }

def load_models(model_dir):
    model_info = {}
    for filename in os.listdir(model_dir):
        if filename.endswith(".onnx"):
            model_name = os.path.splitext(filename)[0]
            model_path = os.path.join(model_dir, filename)
            model_info[model_name] = load_model_info(model_path)
    return model_info

# Transformation
def _is_pil_image(img):
    if isinstance(img, Image.Image):
        return True
    return False


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def to_tensor(pic):
    if not(_is_pil_image(pic) or _is_numpy_image(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if isinstance(pic, np.ndarray):
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        return img.float()
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float()
    else:
        return img


def inference_models(batch, model_info, bboxes):
    batch_size = len(batch)
    num_models = len(model_info)
    num_classes = None

    combined_logits_sum = None

    for model_name, model in model_info.items():
        session = model["session"]
        input_name = model["input_name"]
        output_name = model["output_name"]

        h_input, w_input, model_type, scale = parse_model_name(model_name)
        params = [{"org_img": img, "bbox": bbox, "scale": scale, "out_w": w_input, "out_h": h_input, "crop": True if scale is not None else False} for img, bbox in zip(batch, bboxes)]
        cropped_images = image_cropper.crop_batch(batch, params)

        imgs_tensor = test_transform(cropped_images)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        imgs_tensor = imgs_tensor.to(device)

        imgs_array = imgs_tensor.cpu().numpy()

        result = session.run([output_name], {input_name: imgs_array})

        result_array = np.array(result[0])

        softmax_result = torch.softmax(torch.tensor(result_array), dim=1).cpu().numpy()

        if combined_logits_sum is None:
            num_classes = result_array.shape[1]
            combined_logits_sum = np.zeros((batch_size, num_classes))

        combined_logits_sum += softmax_result

    combined_logits_avg = combined_logits_sum / num_models

    return combined_logits_avg

def test_transform(images):
    transformed_images = [to_tensor(img) for img in images]
    return torch.stack(transformed_images)

def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True
    
def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            filenames.append(filename)
        else:
            print(f"Warning: Unable to load image at {img_path}")
    return images, filenames

def decode_base64_img(base64_string):
    # Decode the Base64 string
    decoded_bytes = base64.b64decode(base64_string)
    
    # Convert bytes to numpy array
    nparr = np.frombuffer(decoded_bytes, np.uint8)
    
    # Decode numpy array to image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return image

@app.route('/predict', methods=['POST'])
def predict():

    # images, filenames = load_images_from_folder(args.image_dir)
    data = request.json
    base64_images = data['images']

    images = []
    for base64_img in base64_images:
        images.append(decode_base64_img(base64_img))

    all_results = []

    for i in range(0, len(images), args.batch_size):
        batch = images[i:i+args.batch_size]
        bboxes = model_test.get_bboxes(batch)

        result_batch = inference_models(batch, model_info, bboxes)
        all_results.extend(result_batch)
    
    total = len(all_results)
    real = 0
    fake = 0
    for result in all_results:
        liveness_score = result[1]
        if(liveness_score >= args.model_threshold):
            real += 1
        else:
            fake += 1

    voting_score = real / total
    if(voting_score >= args.voting_threshold):
        return jsonify({"result": "Real", "voting_score": voting_score})
    else:
        return jsonify({"result": "Fake", "voting_score": voting_score})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images and perform anti-spoofing detection.")
    # parser.add_argument("--image_dir", type=str, default="images/dataset_new/b", help="Directory containing images")
    parser.add_argument("--model_dir", type=str, default="../onnx/dynamic_onnx", help="Directory containing ONNX models")
    parser.add_argument("--model_threshold", type=float, default=0.75, help="Threshold for considering a prediction as real")
    parser.add_argument("--voting_threshold", type=float, default=0.8, help="Threshold for considering the overall result as real")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing images")
    parser.add_argument("--device_id", type=int, default=0, help="Device ID for CUDA")
    args = parser.parse_args()
    
    model_test = AntiSpoofPredict(args.device_id)
    image_cropper = CropImage()
    model_info = load_models(args.model_dir)
    
    app.run(debug=True)
