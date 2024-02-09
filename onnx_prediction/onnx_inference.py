import pickle
import numpy as np
import onnxruntime
import torch
'''
# Load the saved PyTorch tensor data from the pickle file
with open('tensor_data.pkl', 'rb') as f:
    img_numpy = pickle.load(f)

# Convert the NumPy array back to a PyTorch tensor
img_tensor = torch.tensor(img_numpy)
'''
import os
import cv2
import math
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageEnhance
import numbers
import types
import collections
import warnings


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

if __name__ == "__main__":
    # Define the path to the image
    image_path = '/home/dev/Documents/Silent-Face-Anti-Spoofing/img_cropped_4_0_0_80x80_MiniFASNetV1SE.jpg'

    # Load the image using OpenCV
    img = cv2.imread(image_path)

    # Apply the transformation
    img_tensor = test_transform(img)




    # Move the tensor to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)

    # Convert the PyTorch tensor to a NumPy array
    input_tensor = img_tensor.cpu().numpy()

    # Initialize ONNX runtime session and load the model
    onnx_model_path = "/home/dev/Documents/Silent-Face-Anti-Spoofing/onnx/dynamic_model2.onnx"
    session = onnxruntime.InferenceSession(onnx_model_path)

    # Get input and output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Perform inference
    result = session.run([output_name], {input_name: input_tensor})

    # Convert the result to a PyTorch tensor
    result_tensor = torch.tensor(result[0])

    # Apply softmax to the result tensor along the specified dimension (usually the class dimension)
    softmax_result = torch.softmax(result_tensor, dim=1)

    # Convert softmax result back to NumPy array
    softmax_numpy = softmax_result.cpu().detach().numpy()

    # Print the inference result
    print("Softmax result:", softmax_numpy)
