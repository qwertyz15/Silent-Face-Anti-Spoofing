import os
import cv2
import math
import torch
import numpy as np
import torch.nn.functional as F


from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2,MiniFASNetV1SE,MiniFASNetV2SE
from src.data_io import transform as trans
from src.utility import get_kernel, parse_model_name

class Detection:
    def __init__(self):
        caffemodel = "./resources/detection_model/Widerface-RetinaFace.caffemodel"
        deploy = "./resources/detection_model/deploy.prototxt"
        self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
        self.detector_confidence = 0.6

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
        

'''
class Detection:
    def __init__(self):
        caffemodel = "./resources/detection_model/Widerface-RetinaFace.caffemodel"
        deploy = "./resources/detection_model/deploy.prototxt"
        self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
        self.detector_confidence = 0.6

    def get_bboxes(self, images, batch_size):
        bboxes = []
        processed_images = []

        for img in images:
            height, width = img.shape[0], img.shape[1]
            aspect_ratio = width / height
            if img.shape[1] * img.shape[0] >= 192 * 192:
                img = cv2.resize(img,
                                 (int(192 * math.sqrt(aspect_ratio)),
                                  int(192 / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)
            processed_images.append(img)

        # Perform batch inference
        num_batches = math.ceil(len(processed_images) / batch_size)
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(processed_images))
            batch_images = processed_images[start_idx:end_idx]

            blob = cv2.dnn.blobFromImages(batch_images, 1, mean=(104, 117, 123))
            self.detector.setInput(blob, 'data')
            out = self.detector.forward('detection_out')

            for batch_out in out:
                for detection in batch_out:
                    confidence = detection[2]
                    if confidence > self.detector_confidence:
                        left, top, right, bottom = detection[3] * width, detection[4] * height, \
                                                   detection[5] * width, detection[6] * height
                        bbox = [int(left), int(top), int(right - left + 1), int(bottom - top + 1)]
                        bboxes.append(bbox)

        return bboxes
        
'''
