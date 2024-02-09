# -*- coding: utf-8 -*-
# @Time : 20-6-4 上午9:59
# @Author : zhuying
# @Company : Minivision
# @File : train_main.py
# @Software : PyCharm

import os
from collections import OrderedDict
import copy
import torch
from torch import optim
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm
from tensorboardX import SummaryWriter

from src.utility import get_time, extract_model_type
from src.model_lib.MultiFTNet import MultiFTNet
from src.model_lib.MiniFASNet import MiniFASNetV2 
from src.data_io.dataset_loader import get_train_loader, get_val_loader
import copy


class TrainMain:
    def __init__(self, conf):
        self.conf = conf
        print(self.conf)
        self.board_loss_every = conf.board_loss_every
        self.save_every = conf.save_every
        self.step = 0
        self.start_epoch = 0
        self.train_loader = get_train_loader(self.conf)
        self.val_loader = get_val_loader(self.conf) 

    def train_model(self):
        self._init_model_param()
        self._train_stage()

    def _init_model_param(self):
        self.cls_criterion = CrossEntropyLoss()
        self.ft_criterion = MSELoss()
        self.model = self._define_network()
        self.optimizer = optim.SGD(self.model.module.parameters(),
                                   lr=self.conf.lr,
                                   weight_decay=5e-4,
                                   momentum=self.conf.momentum)

        self.schedule_lr = optim.lr_scheduler.MultiStepLR(
            self.optimizer, self.conf.milestones, self.conf.gamma, - 1)

        print("lr: ", self.conf.lr)
        print("epochs: ", self.conf.epochs)
        print("milestones: ", self.conf.milestones)


    def _train_stage(self):
        self.writer = SummaryWriter(self.conf.log_path)
        self.best_val_accuracy = float('-inf')
        self.best_val_loss = float('inf')
        for e in range(self.start_epoch, self.conf.epochs):
            print(f"Epoch {e} started")

            # Training phase
            self.model.train()
            running_loss = 0.
            running_acc = 0.
            for sample, ft_sample, target in tqdm(iter(self.train_loader)):
                imgs = [sample, ft_sample]
                labels = target
                loss, acc, _, _ = self._train_batch_data(imgs, labels)
                running_loss += loss
                running_acc += acc

            train_loss = running_loss / len(self.train_loader)
            train_acc = running_acc / len(self.train_loader)

            # Convert to standard Python number if necessary
            if isinstance(train_loss, torch.Tensor):
                train_loss = train_loss.item()
            if isinstance(train_acc, torch.Tensor):
                train_acc = train_acc.item()

            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

            # # Validation phase
            # self.model.eval()  # Set the model to evaluation mode
            # running_val_loss = 0.0
            # running_val_accuracy = 0.0
            # with torch.no_grad():  # Disable gradient computation
            #     for val_sample, val_ft_sample, val_target in tqdm(iter(self.val_loader)):
            #         val_imgs = [val_sample, val_ft_sample]
            #         val_labels = val_target
            #         val_loss, val_accuracy = self._validate_batch_data(val_imgs, val_labels)
            #         running_val_loss += val_loss
            #         running_val_accuracy += val_accuracy

            # val_loss = running_val_loss / len(self.val_loader)
            # val_accuracy = running_val_accuracy / len(self.val_loader)

            # # Convert to standard Python number if necessary
            # if isinstance(val_loss, torch.Tensor):
            #     val_loss = val_loss.item()
            # if isinstance(val_accuracy, torch.Tensor):
            #     val_accuracy = val_accuracy.item()

            # print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

            # Validation phase
            self.model.eval()  # Set the model to evaluation mode
            running_val_loss = 0.0
            running_val_accuracy = 0.0
            valid_batches = 0  # Counter for valid batches
            _skip = 9
            _count = 0


            with torch.no_grad():  # Disable gradient computation
                # for i, data in enumerate(tqdm(iter(self.val_loader))):
                for i, data in enumerate(tqdm(iter(self.val_loader))):
                    try:
                        val_sample, val_ft_sample, val_target = data
                        val_imgs = [val_sample, val_ft_sample]
                        val_labels = val_target
                        val_loss, val_accuracy = self._validate_batch_data(val_imgs, val_labels)
                        running_val_loss += val_loss
                        running_val_accuracy += val_accuracy
                        valid_batches += 1

                    except Exception as e:
                        print(f"Skipping batch number {i} due to an error: {e}")
                        continue  # Skip this batch and continue to the next one

            # Calculate average loss and accuracy over valid batches
            val_loss = running_val_loss / valid_batches if valid_batches > 0 else 0
            val_accuracy = running_val_accuracy / valid_batches if valid_batches > 0 else 0

            if (val_accuracy > self.best_val_accuracy):
                self.best_val_accuracy = val_accuracy
                self.best_model_state = copy.deepcopy(self.model.state_dict())

            # Convert to standard Python number if necessary
            if isinstance(val_loss, torch.Tensor):
                val_loss = val_loss.item()
            if isinstance(val_accuracy, torch.Tensor):
                val_accuracy = val_accuracy.item()

            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")


            # Update learning rate and save model
            self.schedule_lr.step()
            lr = self.optimizer.param_groups[0]['lr']
            print("Learning Rate: ", lr)
            if self.conf.save_every and (e % self.conf.save_every == 0 or e == self.conf.epochs - 1):
                time_stamp = get_time()
                self._save_state(time_stamp, extra=self.conf.job_name)

        self.writer.close()

    
    def _validate_batch_data(self, imgs, labels):
        # Move labels to the same device as the model
        labels = labels.to(self.conf.device)

        # Forward pass: compute predicted outputs by passing inputs to the model
        outputs = self.model(imgs[0].to(self.conf.device))

        # Calculate the batch's loss
        # Assuming the outputs are logits and the labels are class indices
        loss_cls = self.cls_criterion(outputs, labels)

        # Calculate the batch's accuracy
        acc = self._get_accuracy(outputs, labels)[0]

        # Return loss and accuracy
        return loss_cls.item(), acc




    def _train_batch_data(self, imgs, labels):
        self.optimizer.zero_grad()
        labels = labels.to(self.conf.device)
        embeddings, feature_map = self.model.forward(imgs[0].to(self.conf.device))

        loss_cls = self.cls_criterion(embeddings, labels)
        loss_fea = self.ft_criterion(feature_map, imgs[1].to(self.conf.device))

        loss = 0.5*loss_cls + 0.5*loss_fea
        acc = self._get_accuracy(embeddings, labels)[0]
        loss.backward()
        self.optimizer.step()
        return loss.item(), acc, loss_cls.item(), loss_fea.item()

    @staticmethod
    def print_model_weights(model):
        for name, param in model.state_dict().items():
            print(f"Layer: {name}")
            print(f"Values: \n{param}\n")



  
    def _define_network(self):
        param = {
            'num_classes': self.conf.num_classes,
            'img_channel': self.conf.input_channel,
            'embedding_size': self.conf.embedding_size,
            'conv6_kernel': self.conf.kernel_size
        }

        # Ensure pretrained_model_path is provided
        pretrained_model_path = self.conf.pretrained_model_path
        if not pretrained_model_path:
            raise ValueError("Pretrained model path must be provided")

        # Initialize the model
        model = MultiFTNet(**param).to(self.conf.device)

        # Load pretrained weights
        state_dict = torch.load(pretrained_model_path, map_location=self.conf.device)

        # Adjust for the DataParallel wrapper if necessary
        if list(state_dict.keys())[0].startswith('module.'):
            new_state_dict = {k[7:]: v for k, v in state_dict.items()}
        else:
            new_state_dict = state_dict

        model.load_state_dict(new_state_dict, strict=False)

        # Wrap with DataParallel if multiple devices are available
        model = torch.nn.DataParallel(model, self.conf.devices)

        model.to(self.conf.device)

        # Optional: Print model weights for verification
        self.print_model_weights(model)

        return model



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

    # def _save_state(self, time_stamp, extra=None):
    #     save_path = self.conf.model_path
    #     torch.save(self.model.state_dict(), save_path + '/' +
    #                ('{}_{}_model_iter-{}.pth'.format(time_stamp, extra, self.step)))

    def _save_state(self, time_stamp, extra=None):
        save_path = self.conf.model_path

        # Save the current model
        torch.save(self.model.state_dict(), save_path + '/' +
                ('{}_{}_model_iter-{}.pth'.format(time_stamp, extra, self.step)))

        # Save the best model if it exists
        if hasattr(self, 'best_model_state'):
            best_model_path = save_path + '/' + 'best.pth'
            torch.save(self.best_model_state, best_model_path)

        print("Models saved successfully.")                   
                   
                   
