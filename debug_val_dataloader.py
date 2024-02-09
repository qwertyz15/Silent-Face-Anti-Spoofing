import os
import torch
from src.data_io.dataset_loader import get_val_loader
from torch.utils.data import DataLoader

# Simple class to convert dictionary to object
class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

# Function to test individual samples in the dataset
def test_dataset_samples(dataset, num_samples=10):
    for i in range(num_samples):
        try:
            # Accessing individual samples directly from the dataset
            data = dataset.dataset[i]
            print(f"Sample {i}: Success")
        except Exception as e:
            print(f"Error with sample {i}: {e}")

# Default configuration values
conf_dict = {
    'lr': 0.1, 
    'milestones': [10, 15, 22], 
    'gamma': 0.1, 
    'epochs': 25, 
    'momentum': 0.9, 
    'batch_size': 1024, 
    'num_classes': 2, 
    'input_channel': 3, 
    'embedding_size': 128, 
    'train_root_path': './datasets/rgb_image', 
    'val_root_path': './datasets/rgb_image_val', 
    'snapshot_dir_path': './saved_logs/snapshot', 
    'log_path': './saved_logs/jobs/Anti_Spoofing_2.7_80x80/Jan19_17-30-52', 
    'board_loss_every': 10, 
    'save_every': 30, 
    'devices': [0], 
    'patch_info': '2.7_80x80', 
    'pretrained_model_path': '/home/dev/Documents/Silent-Face-Anti-Spoofing/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth', 
    'input_size': [80, 80], 
    'kernel_size': [5, 5], 
    'device': 'cuda:0', 
    'ft_height': 10, 
    'ft_width': 10, 
    'model_path': './saved_logs/snapshot/Anti_Spoofing_2.7_80x80', 
    'job_name': 'Anti_Spoofing_2.7_80x80'
}

# Convert dictionary to object
conf = Config(**conf_dict)

# Load your validation dataset
valset = get_val_loader(conf)

# Test first few samples directly from the dataset
print("Testing individual samples in the dataset:")
test_dataset_samples(valset, num_samples=10)

# Simplify the DataLoader for debugging
val_loader = DataLoader(
    valset,
    batch_size=conf.batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=0)  # Set to 0 for debugging

# Test DataLoader
# Test DataLoader
print("\nTesting DataLoader:")
for i, batch in enumerate(val_loader):
    if i == 9:  # Skipping the 10th batch (since indexing starts at 0)
        print("Skipping batch 10")
        continue

    print(f"Processing batch {i+1}:")
    try:
        # Process the batch
        val_sample, val_ft_sample, val_target = batch
        print("Batch processed successfully")
    except Exception as e:
        print(f"Error processing batch {i+1}: {e}")
