# -*- coding: utf-8 -*-
# @Time : 20-6-9 上午10:20
# @Author : zhuying
# @Company : Minivision
# @File : anti_spoof_predict.py
# @Software : PyCharm

import os
import cv2
import math
import torch
import numpy as np
import torch.nn.functional as F


from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2,MiniFASNetV1SE,MiniFASNetV2SE
from src.data_io import transform as trans
from src.utility import get_kernel, parse_model_name

import pickle
import torch

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE':MiniFASNetV1SE,
    'MiniFASNetV2SE':MiniFASNetV2SE
}


class Detection:
    def __init__(self):
        caffemodel = "./resources/detection_model/Widerface-RetinaFace.caffemodel"
        deploy = "./resources/detection_model/deploy.prototxt"
        self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
        self.detector_confidence = 0.6

    def get_bbox(self, img):
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img = cv2.resize(img,
                             (int(192 * math.sqrt(aspect_ratio)),
                              int(192 / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)

        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        max_conf_index = np.argmax(out[:, 2])
        left, top, right, bottom = out[max_conf_index, 3]*width, out[max_conf_index, 4]*height, \
                                   out[max_conf_index, 5]*width, out[max_conf_index, 6]*height
        bbox = [int(left), int(top), int(right-left+1), int(bottom-top+1)]
        return bbox

    def get_bboxes(self, images):
        bboxes = []
        for img in images:
            if img.size == 0:
                print("Empty image array encountered.")
                continue
            height, width = img.shape[:2]
            aspect_ratio = width / height
            resized_img = img
            if img.shape[1] * img.shape[0] >= 192 * 192:
                resized_img = cv2.resize(img, (int(192 * math.sqrt(aspect_ratio)), int(192 / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)
            
            blob = cv2.dnn.blobFromImage(resized_img, 1, mean=(104, 117, 123))
            self.detector.setInput(blob, 'data')
            out = self.detector.forward('detection_out').squeeze()
            max_conf_index = np.argmax(out[:, 2])
            left, top, right, bottom = out[max_conf_index, 3]*width, out[max_conf_index, 4]*height, out[max_conf_index, 5]*width, out[max_conf_index, 6]*height
            bbox = [int(left), int(top), int(right-left+1), int(bottom-top+1)]
            bboxes.append(bbox)
        return bboxes

class AntiSpoofPredict(Detection):
    def __init__(self, device_id):
        super(AntiSpoofPredict, self).__init__()
        self.device = torch.device("cuda:{}".format(device_id)
                                   if torch.cuda.is_available() else "cpu")

    def _load_model(self, model_path):
        # define model
        model_name = os.path.basename(model_path)
        h_input, w_input, model_type, _ = parse_model_name(model_name)
        self.kernel_size = get_kernel(h_input, w_input,)
        self.model = MODEL_MAPPING[model_type](conv6_kernel=self.kernel_size).to(self.device)

        # load model weight
        state_dict = torch.load(model_path, map_location=self.device)
        keys = iter(state_dict)
        first_layer_name = keys.__next__()
        if first_layer_name.find('module.') >= 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                new_state_dict[name_key] = value
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)
        return self.model

    # # def predict(self, img, model_path):
    # def predict(self, img, smodel):
    #     test_transform = trans.Compose([
    #         trans.ToTensor(),
    #     ])
    #     img = test_transform(img)
    #     img = img.unsqueeze(0).to(self.device)
        
    #     # Assuming img is your PyTorch tensor
    #     img_numpy = img.cpu().numpy()  # Convert PyTorch tensor to NumPy array
    #     # Now you can save the NumPy array using pickle
    #     with open('tensor_data.pkl', 'wb') as f:
    #         pickle.dump(img_numpy, f)
        
    #     # self._load_model(model_path)
    #     self.model = smodel
    #     # self.model.eval()
    #     with torch.no_grad():
    #         result = self.model.forward(img)
    #         result = F.softmax(result).cpu().numpy()
    #     return result

    # # def predict_batch(self, img_batch, model_path):
    # def predict_batch(self, img_batch, smodel):
    #     test_transform = trans.Compose([
    #         trans.ToTensor(),
    #     ])
    #     img_batch = torch.stack([test_transform(img) for img in img_batch])
    #     img_batch = img_batch.to(self.device)
    #     # self._load_model(model_path)
    #     self.model = smodel
    #     # self.model.eval()
    #     with torch.no_grad():
    #         result = self.model.forward(img_batch)
    #         result = F.softmax(result, dim=1).cpu().numpy()
    #     return result

    def preprocess_img(self, img):
        test_transform = trans.Compose([
            trans.ToTensor(),
        ])
        img = test_transform(img)
        img = img.unsqueeze(0).to(self.device)
        return img

    def preprocess_batch(self, img_batch):
        test_transform = trans.Compose([
            trans.ToTensor(),
        ])
        img_batch = torch.stack([test_transform(img) for img in img_batch])
        img_batch = img_batch.to(self.device)
        return img_batch

    def predict(self, img, smodel):
        # Preprocess the image
        img = self.preprocess_img(img)

        # Perform inference
        self.model = smodel
        with torch.no_grad():
            result = self.model.forward(img)
            result = F.softmax(result, dim=1).cpu().numpy()
        return result

    def predict_batch(self, img_batch, smodel):
        # Preprocess the batch of images
        img_batch = self.preprocess_batch(img_batch)

        # Perform batch inference
        self.model = smodel
        with torch.no_grad():
            result = self.model.forward(img_batch)
            result = F.softmax(result, dim=1).cpu().numpy()
        return result

    def export_to_onnx(self, model_path, onnx_path, input_size=(1, 3, 80, 80)):
        self._load_model(model_path)
        dummy_input = torch.randn(*input_size).to(self.device)
        torch.onnx.export(self.model, dummy_input, onnx_path, export_params=True, opset_version=11,
                          do_constant_folding=True, input_names=['input'], output_names=['output'])


    def export_to_onnx_dynamic(self, model_path, onnx_path, input_size=(1, 3, 224, 224)):
        self._load_model(model_path)
        
        # Modify the input size to have a dynamic batch dimension
        dynamic_input_size = (None, *input_size[1:])
        dummy_input = torch.randn(1, *input_size[1:]).to(self.device)
        
        torch.onnx.export(
            self.model, 
            dummy_input, 
            onnx_path, 
            export_params=True, 
            opset_version=11,
            do_constant_folding=True, 
            input_names=['input'], 
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},  # Make batch size dynamic
                'output': {0: 'batch_size'}
            }
        )
        print(f"Model exported to ONNX format at {onnx_path}")
