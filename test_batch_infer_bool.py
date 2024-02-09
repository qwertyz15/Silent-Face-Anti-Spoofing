import os
import cv2
import numpy as np
import argparse
import time
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name


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


def batch_test(image_folder, model_dir, device_id, batch_size, threshold):
    
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
    print(f"Model loading time: {time.time() - st1} seconds")
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
            # value = prediction[0][label] / 2
            liveness_score = prediction[1] / 2

            if liveness_score >= threshold:
                label = 1
            else:
                label = 0
            # value = prediction[label] / 2
            result = ("Real" if label == 1 else "Fake") + f" Score: {liveness_score:.2f}"
            results.append((batch_filenames[j], result))
    print(f"Model Inference Time: {time.time() - st2} seconds [{(len(results))} images]")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Test")
    parser.add_argument("--device_id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--model_dir", type=str, default="./resources/anti_spoof_models", help="Path to the model directory")
    parser.add_argument("--image_folder", type=str, required=True, help="Folder containing images for testing")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for testing")
    parser.add_argument("--model_threshold", type=int, default=0.75, help="liveness threshold")
    parser.add_argument("--voting_threshold", type=int, default=0.8, help="frames threshold")

    args = parser.parse_args()
    result_map = batch_test(args.image_folder, args.model_dir, args.device_id, args.batch_size, args.model_threshold)
    total_image = len(result_map)
    total_real = 0
    total_fake = 0
    for image_name, result in result_map:
        if("Real Score" in result):
            total_real += 1
        else:
            total_fake += 1

        # print(f"{image_name}: {result}")
    liveness = total_real/total_image
    if(liveness >= args.voting_threshold):
        print("Real")
    else:
        print("Fake")
