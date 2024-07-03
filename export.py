# -*- coding: utf-8 -*-
# @Time : 20-6-9 上午10:20
# @Author : zhuying
# @Company : Minivision
# @File : export.py
# @Software : PyCharm

import os
import argparse
import torch
from src.anti_spoof_predict import AntiSpoofPredict

def export_model(model_path, onnx_path, device_id):
    anti_spoof = AntiSpoofPredict(device_id=device_id)
    input_size = (1, 3, 80, 80)  # Update input size if necessary
    anti_spoof.export_to_onnx(model_path, onnx_path, input_size)
    print(f"Model exported to ONNX format at {onnx_path}")

if __name__ == "__main__":
    desc = "Export model to ONNX format"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="Which GPU ID to use, [0/1/2/3]. Default is 0")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model file (.pth)")
    parser.add_argument(
        "--onnx_path",
        type=str,
        required=True,
        help="Path where the ONNX model will be saved")
    args = parser.parse_args()
    export_model(args.model_path, args.onnx_path, args.device_id)
