import cv2
import numpy as np
import argparse
import warnings
import time
import os

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')

def check_image(image):
    height, width, channel = image.shape
    if width / height != 3 / 4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True

def test_video(video_path, model_dir, device_id, threshold):
    cap = cv2.VideoCapture(video_path)
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break



        image_bbox = model_test.get_bbox(frame)
        prediction = np.zeros((1, 3))
        test_speed = 0

        all_models = dict()
        for model_name in os.listdir(model_dir):
            model_path = os.path.join(model_dir, model_name)
            smodel = model_test._load_model(model_path)
            smodel.eval()
            all_models[model_name] = smodel

        for model_name in os.listdir(model_dir):
            smodel = all_models[model_name]
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": frame,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = image_cropper.crop(**param)
            start = time.time()
            # prediction += model_test.predict(img, os.path.join(model_dir, model_name))
            prediction += model_test.predict(img, smodel)
            test_speed += time.time() - start

        label = np.argmax(prediction)
        value = prediction[0][label] / 2
        liveness_score = prediction[0][1] / 2

        if liveness_score >= threshold:
            label = 1
        else:
            label = 0

        if label == 1:
            print("Real Face. LivenessScore: {:.2f}.".format(liveness_score))
            result_text = "RealFace | LivenessScore: {:.2f}".format(liveness_score)
            color = (0, 255, 0)
        else:
            print("Fake Face. LivenessScore: {:.2f}.".format(liveness_score))
            result_text = "FakeFace | LivenessScore: {:.2f}".format(liveness_score)
            color = (0, 0, 255)

        print("Prediction cost {:.2f} s".format(test_speed))
        cv2.rectangle(
            frame,
            (image_bbox[0], image_bbox[1]),
            (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
            color, 2)
        cv2.putText(
            frame,
            result_text,
            (image_bbox[0], image_bbox[1] - 5),
            cv2.FONT_HERSHEY_COMPLEX, 0.5 * frame.shape[0] / 1024, color)

        cv2.imshow("Result", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    def check_zero_to_one(value):
        fvalue = float(value)
        if fvalue <= 0 or fvalue >= 1:
            raise argparse.ArgumentTypeError("%s is an invalid value" % value)
        return fvalue

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
        "--video_path", "-v",
        type=str,
        default=0,
        help="video used to test")
    parser.add_argument(
        "--threshold", "-t",
        type=check_zero_to_one,
        default=0.6,
        help="liveness threshold")
    args = parser.parse_args()

    test_video(args.video_path, args.model_dir, args.device_id, args.threshold)
