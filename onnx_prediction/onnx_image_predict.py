import os
import cv2
import sys
import numpy as np
import torch
import onnxruntime
import argparse
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

def inference_models(img, model_info, image_bbox):
    combined_logits = None
    for model_name, model in model_info.items():
        session = model["session"]
        input_name = model["input_name"]
        output_name = model["output_name"]

        # Preprocessing Image
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        print(h_input, w_input, model_type, scale)
        param = {
            "org_img": img,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        # Apply the transformation
        img_tensor = test_transform(img)

        # Move the tensor to CUDA if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(device)

        # Convert the PyTorch tensor to a NumPy array
        img_tensor = img_tensor.cpu().numpy()


        # Perform inference
        result = session.run([output_name], {input_name: img_tensor})

        # Convert the result to a PyTorch tensor
        result_tensor = torch.tensor(result[0])

        # Apply softmax to the result tensor along the specified dimension (usually the class dimension)
        softmax_result = torch.softmax(result_tensor, dim=1)

        # Convert softmax result back to NumPy array
        softmax_numpy = softmax_result.cpu().detach().numpy()
        print(softmax_numpy)

        if combined_logits is None:
            combined_logits = softmax_numpy
        else:
            combined_logits += softmax_numpy

    # Average the logits from all models and divide by 2
    combined_logits /= len(model_info)

    return combined_logits
    

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


def test_transform(img):
    return to_tensor(img)


#############################33    
def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform anti-spoofing detection on a single image.")
    parser.add_argument("--threshold", type=float, default=0.75, help="Threshold for considering a prediction as real")
    parser.add_argument("--image_path", type=str, default='../images/sample/image_F2.jpg', help="Path to the input image")
    parser.add_argument("--model_dir", type=str, default="/home/dev/Documents/Silent-Face-Anti-Spoofing/onnx/dynamic_onnx", help="Directory containing ONNX models")
    args = parser.parse_args()

    threshold = args.threshold
    image_path = args.image_path
    model_dir = args.model_dir

    model_test = AntiSpoofPredict(0)
    image_cropper = CropImage()
    image = cv2.imread(image_path)
    frame = image
    result = check_image(image)

    if result is False:
        sys.exit() 
    image_bbox = model_test.get_bbox(image)
    print(image_bbox)

    # Load models from the directory
    model_info = load_models(model_dir)
    print(model_info)

    # Perform inference with multiple models
    combined_softmax = inference_models(image, model_info, image_bbox)

    # Print the combined softmax result
    print("Combined softmax result:", combined_softmax)
    prediction = combined_softmax

    label = np.argmax(prediction)
    liveness_score = prediction[0][1] 

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

    # Draw rectangle and text on the frame
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

    # Display the result
    cv2.imshow("Result", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
