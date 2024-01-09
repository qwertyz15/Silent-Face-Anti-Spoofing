import os
import cv2
import numpy as np
import argparse
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def load_images_from_subfolders(parent_folder):
    all_images = []
    all_labels = []
    for subfolder in os.listdir(parent_folder):
        if os.path.isdir(os.path.join(parent_folder, subfolder)):
            label = int(subfolder)
            folder_path = os.path.join(parent_folder, subfolder)
            images, _ = load_images_from_folder(folder_path)
            labels = [label] * len(images)
            all_images.extend(images)
            all_labels.extend(labels)
    return all_images, all_labels

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images, [filename for filename in os.listdir(folder)]

def validate_model(validation_folder, model_dir, device_id, batch_size):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    images, true_labels = load_images_from_subfolders(validation_folder)
    predicted_labels = []
    file_names = []

    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        true_batch_labels = true_labels[i:i+batch_size]
        bboxes = model_test.get_bboxes(batch)

        batch_predictions = np.zeros((len(batch), 3))

        for model_name in os.listdir(model_dir):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            params = [{"org_img": img, "bbox": bbox, "scale": scale, "out_w": w_input, "out_h": h_input, "crop": True if scale is not None else False} for img, bbox in zip(batch, bboxes)]
            cropped_images = image_cropper.crop_batch(batch, params)
            batch_predictions += model_test.predict_batch(cropped_images, os.path.join(model_dir, model_name))

        for j, prediction in enumerate(batch_predictions):
            label = np.argmax(prediction)
            predicted_labels.append(label)
            file_names.append(f"Batch {i//batch_size + 1} Image {j + 1}")  # Update this to reflect actual file names

    accuracy = accuracy_score(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels, zero_division=0)  # Added zero_division parameter
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    return accuracy, report, zip(file_names, predicted_labels), conf_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validation")
    parser.add_argument("--device_id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--model_dir", type=str, default="./resources/anti_spoof_models", help="Path to the model directory")
    parser.add_argument("--validation_folder", type=str, required=True, help="Folder containing validation images")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for validation")

    args = parser.parse_args()
    accuracy, report, results, conf_matrix = validate_model(args.validation_folder, args.model_dir, args.device_id, args.batch_size)

    print(f"Validation Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(conf_matrix)
    # for image_name, prediction in results:
    #    print(f"{image_name}: {'Real' if prediction == 1 else 'Fake'}")

