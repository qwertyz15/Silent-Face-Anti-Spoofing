import os
import cv2
import numpy as np
import argparse
import time
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

# def load_images_from_folder(folder):
#     images = []
#     filenames = []
#     for filename in os.listdir(folder):
#         img = cv2.imread(os.path.join(folder, filename))
#         if img is not None:
#             images.append(img)
#             filenames.append(filename)
#     return images, filenames

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

# def batch_test(image_folder, model_dir, device_id, batch_size):
#     model_test = AntiSpoofPredict(device_id)
#     image_cropper = CropImage()
#     images, filenames = load_images_from_folder(image_folder)
#     results = []

#     for i in range(0, len(images), batch_size):
#         batch = images[i:i+batch_size]
#         batch_filenames = filenames[i:i+batch_size]
#         batch_predictions = np.zeros((len(batch), 3))

#         for model_name in os.listdir(model_dir):
#             h_input, w_input, model_type, scale = parse_model_name(model_name)
#             cropped_images = []

#             for img in batch:
#                 param = {
#                     "org_img": img,
#                     "bbox": model_test.get_bbox(img),
#                     "scale": scale,
#                     "out_w": w_input,
#                     "out_h": h_input,
#                     "crop": True if scale is not None else False,
#                 }
#                 cropped_img = image_cropper.crop(**param)
#                 cropped_images.append(cropped_img)

#             batch_predictions += model_test.predict_batch(cropped_images, os.path.join(model_dir, model_name))

#         for j, prediction in enumerate(batch_predictions):
#             label = np.argmax(prediction)
#             value = prediction[label] / 2
#             result = ("Real" if label == 1 else "Fake") + f" Score: {value:.2f}"
#             results.append((batch_filenames[j], result))

#     return results

def batch_test(image_folder, model_dir, device_id, batch_size):
    
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()

    
    images, filenames = load_images_from_folder(image_folder)
    results = []

    all_models = dict()
    st1 = time.time()
    for model_name in os.listdir(model_dir):
        model_path = os.path.join(model_dir, model_name)
        smodel = model_test._load_model(model_path)
        smodel.eval()
        all_models[model_name] = smodel
    print(time.time() - st1)
    st2 = time.time()
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch_filenames = filenames[i:i+batch_size]
        batch_predictions = np.zeros((len(batch), 3))
        bboxes = model_test.get_bboxes(batch)  # Get bboxes for batch

        for model_name in os.listdir(model_dir):
            smodel = all_models[model_name]
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            params = [{"org_img": img, "bbox": bbox, "scale": scale, "out_w": w_input, "out_h": h_input, "crop": True if scale is not None else False} for img, bbox in zip(batch, bboxes)]
            cropped_images = image_cropper.crop_batch(batch, params)

            # batch_predictions += model_test.predict_batch(cropped_images, os.path.join(model_dir, model_name))
            batch_predictions += model_test.predict_batch(cropped_images, smodel)


        for j, prediction in enumerate(batch_predictions):
            label = np.argmax(prediction)
            value = prediction[label] / 2
            result = ("Real" if label == 1 else "Fake") + f" Score: {value:.2f}"
            results.append((batch_filenames[j], result))
    print(time.time() - st2)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Test")
    parser.add_argument("--device_id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--model_dir", type=str, default="./resources/anti_spoof_models", help="Path to the model directory")
    parser.add_argument("--image_folder", type=str, required=True, help="Folder containing images for testing")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for testing")

    args = parser.parse_args()
    result_map = batch_test(args.image_folder, args.model_dir, args.device_id, args.batch_size)
    for image_name, result in result_map:
        print(f"{image_name}: {result}")
