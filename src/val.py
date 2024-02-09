# -*- coding: utf-8 -*-
# @Time : 20-6-4 上午9:59
# @Author : zhuying
# @Company : Minivision
# @File : val.py
# @Software : PyCharm

import os
import argparse
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from src.utility import extract_model_type
from src.model_lib.MultiFTNet import MultiFTNet
from src.data_io.dataset_loader import get_val_loader

class ValidateMain:
    def __init__(self, conf):
        self.conf = conf
        self.model = self._define_network()
        self.val_loader = get_val_loader(self.conf)
        self.cls_criterion = CrossEntropyLoss()

    def _define_network(self):
        param = {
            'num_classes': self.conf.num_classes,
            'img_channel': self.conf.input_channel,
            'embedding_size': self.conf.embedding_size,
            'conv6_kernel': self.conf.kernel_size
        }

        model_type = extract_model_type(self.conf.model_path)
        param['model_type'] = model_type

        model = MultiFTNet(**param).to(self.conf.device)
        model = torch.nn.DataParallel(model, self.conf.devices)

        state_dict = torch.load(self.conf.model_path, map_location=self.conf.device)
        model.load_state_dict(state_dict, strict=True)

        model.eval()
        return model

    def validate_model(self):
        running_val_loss = 0.0
        running_val_accuracy = 0.0

        with torch.no_grad():
            for val_sample, val_ft_sample, val_target in tqdm(iter(self.val_loader)):
                val_imgs = [val_sample, val_ft_sample]
                val_labels = val_target.to(self.conf.device)

                outputs = self.model(val_imgs[0].to(self.conf.device))
                loss_cls = self.cls_criterion(outputs, val_labels)

                acc = self._get_accuracy(outputs, val_labels)[0]

                running_val_loss += loss_cls.item()
                running_val_accuracy += acc

        val_loss = running_val_loss / len(self.val_loader)
        val_accuracy = running_val_accuracy / len(self.val_loader)

        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    def _get_accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        ret = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
            ret.append(correct_k.mul_(1. / batch_size))
        return ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validation Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the validation dataset")
    parser.add_argument("--patch_info", type=str, default="1_80x80", help="Patch info for validation")
    parser.add_argument("--device_id", type=int, default=0, help="GPU device ID to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for validation")
    # Add other necessary arguments and default configuration loading as in train.py
    args = parser.parse_args()

    conf = get_default_config()  # Assuming this function is similar to the one in train.py
    conf.model_path = args.model_path
    conf.dataset_path = args.dataset_path
    conf.patch_info = args.patch_info
    conf.device_id = args.device_id
    conf.batch_size = args.batch_size
    # Update conf with other args if needed

    validator = ValidateMain(conf)
    validator.validate_model()
