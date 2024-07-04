# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import os
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')

# Function to check image dimensions
def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True

# Function to perform the anti-spoofing test
def test(image_name, model_dir, device_id, threshold):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    image = cv2.imread(image_name)
    
    # Check if image dimensions are appropriate
    result = check_image(image)
    if not result:
        return
    
    # Get bounding box for the face in the image
    image_bbox = model_test.get_bbox(image)
    
    # Initialize variables for prediction and time tracking
    prediction = np.zeros((1, 3))
    test_speed = 0
    
    # Iterate through each model in the model directory
    for model_name in os.listdir(model_dir):
        model_path = os.path.join(model_dir, model_name)
        
        # Load the model
        smodel = model_test._load_model(model_path)
        smodel.eval()  # Ensure model is in evaluation mode
        
        # Parse model details
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        
        # Prepare parameters for cropping the image
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        
        # Crop the image
        img = image_cropper.crop(**param)
        
        # Perform prediction using the loaded model
        start = time.time()
        prediction += model_test.predict(img, smodel)
        test_speed += time.time() - start
    
    # Determine the final prediction label based on threshold
    liveness_score = prediction[0][1] / 2
    if liveness_score >= threshold:
        label = 1  # Real Face
    else:
        label = 0  # Fake Face
    
    # Prepare result text and color for annotating the image
    if label == 1:
        print("Image '{}' is Real Face. LivenessScore: {:.2f}.".format(image_name, liveness_score))
        result_text = "RealFace | LivenessScore: {:.2f}".format(liveness_score)
        color = (255, 0, 0)  # Red color
    else:
        print("Image '{}' is Fake Face. LivenessScore: {:.2f}.".format(image_name, liveness_score))
        result_text = "FakeFace | LivenessScore: {:.2f}".format(liveness_score)
        color = (0, 0, 255)  # Blue color
    
    # Print prediction cost
    print("Prediction cost {:.2f} s".format(test_speed))
    
    # Draw bounding box and annotate the image with result
    cv2.rectangle(
        image,
        (image_bbox[0], image_bbox[1]),
        (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
        color, 2)
    cv2.putText(
        image,
        result_text,
        (image_bbox[0], image_bbox[1] - 5),
        cv2.FONT_HERSHEY_COMPLEX, 0.5 * image.shape[0] / 1024, color)
    
    # Save the annotated image
    format_ = os.path.splitext(image_name)[-1]
    result_image_name = image_name.replace(format_, "_result" + format_)
    cv2.imwrite(result_image_name, image)

# Command-line argument parsing
if __name__ == "__main__":
    def check_zero_to_one(value):
        fvalue = float(value)
        if fvalue <= 0 or fvalue >= 1:
            raise argparse.ArgumentTypeError("%s is an invalid value" % value)
        return fvalue
    
    # Define argument parser
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir", "-m",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to test")
    parser.add_argument(
        "--image_name", "-i",
        type=str,
        default="images/sample/image_T1.jpg",
        help="image used to test")
    parser.add_argument(
        "--threshold", "-t",
        type=check_zero_to_one,
        default=0.6,
        help="liveness threshold")
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Call the test function with parsed arguments
    test(args.image_name, args.model_dir, args.device_id, args.threshold)
