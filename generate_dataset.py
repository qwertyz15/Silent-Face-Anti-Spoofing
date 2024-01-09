import os
import cv2
import argparse
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage

def process_image(image_path, device_id, save_path, h_input, w_input, scale):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    image = cv2.imread(image_path)
    image_bbox = model_test.get_bbox(image)

    param = {
        "org_img": image,
        "bbox": image_bbox,
        "scale": scale,
        "out_w": w_input,
        "out_h": h_input,
        "crop": True if scale is not None else False,
    }
    cropped_img = image_cropper.crop(**param)
    cv2.imwrite(save_path, cropped_img)

def generate_scale_dir_name(scale, h_input, w_input):
    if scale is None:
        return f"org_1_{h_input}x{w_input}"
    else:
        scale_str = f"{scale:g}"  # Removes insignificant trailing zeros
        return f"{scale_str}_{h_input}x{w_input}"

def replicate_directory_structure(input_dir, save_dir, scale, h_input, w_input):
    scale_dir_name = generate_scale_dir_name(scale, h_input, w_input)
    for root, _, _ in os.walk(input_dir):
        relative_path = os.path.relpath(root, input_dir)
        save_subdir = os.path.join(save_dir, scale_dir_name, relative_path)
        os.makedirs(save_subdir, exist_ok=True)

def process_directory(input_dir, device_id, save_dir, h_input, w_input, scale):
    replicate_directory_structure(input_dir, save_dir, scale, h_input, w_input)
    
    scale_dir_name = generate_scale_dir_name(scale, h_input, w_input)
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                save_path = os.path.join(save_dir, scale_dir_name, relative_path, file)
                process_image(image_path, device_id, save_path, h_input, w_input, scale)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Cropped Dataset")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory of images to process")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save cropped images")
    parser.add_argument("--device_id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--h_input", type=int, default=80, help="Height of the cropped image")
    parser.add_argument("--w_input", type=int, default=80, help="Width of the cropped image")
    parser.add_argument("--scale", type=float, default=None, help="Scale factor for bounding box adjustment")

    args = parser.parse_args()
    process_directory(args.input_dir, args.device_id, args.save_dir, args.h_input, args.w_input, args.scale)
