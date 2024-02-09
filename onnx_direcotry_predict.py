import os
import cv2
import sys
import numpy as np
import torch
import argparse
import onnxruntime
from PIL import Image
from src.generate_patches import CropImage
from src.utility import parse_model_name
from src.anti_spoof_predict import AntiSpoofPredict

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

def inference_models(batch, model_info, bboxes):
    batch_size = len(batch)
    num_models = len(model_info)
    num_classes = None

    # Initialize the combined logits
    combined_logits_sum = None

    for model_name, model in model_info.items():
        session = model["session"]
        input_name = model["input_name"]
        output_name = model["output_name"]

        # Preprocessing Image
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        params = [{"org_img": img, "bbox": bbox, "scale": scale, "out_w": w_input, "out_h": h_input, "crop": True if scale is not None else False} for img, bbox in zip(batch, bboxes)]
        cropped_images = image_cropper.crop_batch(batch, params)

        # Apply the transformation
        imgs_tensor = test_transform(cropped_images)

        # Move the tensor to CUDA if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        imgs_tensor = imgs_tensor.to(device)

        # Convert the PyTorch tensor to a NumPy array
        imgs_array = imgs_tensor.cpu().numpy()

        # Perform inference
        result = session.run([output_name], {input_name: imgs_array})

        # Convert the result to a NumPy array
        result_array = np.array(result[0])

        # Compute softmax for the result
        softmax_result = torch.softmax(torch.tensor(result_array), dim=1).cpu().numpy()

        # Initialize combined_logits_sum if not already initialized
        if combined_logits_sum is None:
            num_classes = result_array.shape[1]
            combined_logits_sum = np.zeros((batch_size, num_classes))

        # Sum the softmax result to the combined logits
        combined_logits_sum += softmax_result

    # Average the combined logits
    combined_logits_avg = combined_logits_sum / num_models

    return combined_logits_avg
    

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



# def test_transform(images):
#     # Convert the list of images to a NumPy array
#     images_array = np.array(images)

#     # Convert the NumPy array to a PyTorch tensor
#     images_tensor = torch.from_numpy(images_array.transpose((0, 3, 1, 2))).float()

#     return images_tensor

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images and perform anti-spoofing detection.")
    parser.add_argument("--image_dir", type=str, default="../images/dataset_new/b", help="Directory containing images")
    parser.add_argument("--model_dir", type=str, default="../resources/dynamic_onnx", help="Directory containing ONNX models")
    parser.add_argument("--model_threshold", type=float, default=0.75, help="Threshold for considering a prediction as real")
    parser.add_argument("--voting_threshold", type=float, default=0.8, help="Threshold for considering the overall result as real")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing images")
    args = parser.parse_args()

    image_dir = args.image_dir
    model_dir = args.model_dir
    model_threshold = args.model_threshold
    voting_threshold = args.voting_threshold
    batch_size = args.batch_size
    
    model_test = AntiSpoofPredict(0)
    image_cropper = CropImage()

    # Load models from the directory
    model_info = load_models(model_dir)
    # print(model_info)

    images, filenames = load_images_from_folder(image_dir)

    all_results = []  # Store all batch results here

    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch_filenames = filenames[i:i+batch_size]
        bboxes = model_test.get_bboxes(batch)  # Get bboxes for batch

        result_batch = inference_models(batch, model_info, bboxes)
        all_results.extend(result_batch)
    

    total = len(all_results)
    real = 0
    fake = 0
    for result in all_results:
        liveness_score = result[1]
        if(liveness_score >= model_threshold):
            real += 1
        else:
            fake += 1

    voting_score = real / total
    if(voting_score >= voting_threshold):
        print(f"Real: voting_score:{voting_score}")
    else:
        print(f"Fake: voting_score:{voting_score}")
