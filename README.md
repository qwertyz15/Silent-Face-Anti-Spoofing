![Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/images/logo.jpg)  
# Silent-Face-Anti-Spoofing 

This project is Silent-Face-Anti-Spoofing belongs to [minivision technology](https://www.minivision.cn/). You can scan the QR code below to get APK and install it on Android side to experience the effect of real time living detection(silent face anti-spoofing detection).   
<img src="https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/images/静默活体APK.jpeg" width="200" height="200" align=center />  

## Introduction

In this project, we open source the silent face anti-spoofing model with training architecture, data preprocessing method, model training & test script and open source APK for real time testing.  

The main purpose of silent face anti-spoofing detection technology is to judge whether the face in front of the machine is real or fake. The face presented by other media can be defined as false face, including printed paper photos, display screen of electronic products, silicone mask, 3D human image, etc. At present, the mainstream solutions includes cooperative living detection and non cooperative living detection (silent living detection). Cooperative living detection requires the user to complete the specified action according to the prompt, and then carry out the live verification, while the silent live detection directly performs the live verification.  

Since the Fourier spectrum can reflect the difference of true and false faces in frequency domain to a certain extent, we adopt a silent living detection method based on the auxiliary supervision of Fourier spectrum. The model architecture consists of the main classification branch and the auxiliary supervision branch of Fourier spectrum. The overall architecture is shown in the following figure:

![overall architecture](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/images/framework.jpg)  

By using our self-developed model pruning method, the FLOPs of MobileFaceNet is reduced from 0.224G to 0.081G, and the performance of the model is significantly improved (the amount of calculation and parameters is reduced) with little loss of precision.


|Model|FLOPs|Params|
| :------:|:-----:|:-----:| 
|MobileFaceNet|0.224G|0.991M|
|MiniFASNetV1|0.081G|0.414M|
|MiniFASNetV2|0.081G|0.435M|

## APK
### APK source code  
Open source for Android platform deployment code: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing-APK  

### Demo
<img src="https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/images/demo.gif" width="300" height="400"/>  
 
### Performance  
| Model|FLOPs|Speed| FPR | TPR |comments |
| :------:|:-----:|:-----:| :----: | :----: | :----: |
|   APK |84M| 20ms | 1e-5|97.8%| Open Source|
| High precision model |162M| 40ms| 1e-5 |99.7%| Private |

### Test Method 

- Display information: speed(ms), confidence(0 ~ 1) and in living detection test results (true face or false face).
- Click the icon in the upper right corner to set the threshold value. If the confidence level is greater than the threshold value, it is a true face, otherwise it is a fake face.

### Before test you must know

- All the test images must be collected by camera, otherwise it does not conform to the normal scene usage specification, and the algorithm effect cannot be guaranteed.
- Because the robustness of RGB silent living detection depending on camera model and scene, the actual use experience could be different.
- During the test, it should be ensured that a complete face appears in the view, and the rotation angle and vertical direction of the face are less than 30 degrees (in line with the normal face recognition scene), otherwise, the experience will be affected.　

**Tested mobile phone processor**

|type|Kirin990 5G|Kirin990 |Qualcomm845 |Kirin810 |RK3288 |
| :------:|:-----:|:-----:|:-----:|:-----:|:-----:|
|Speed/ms|19|23|24|25|90|

## Repo
### Install dependency Library  
```
pip install -r requirements.txt
```
### Clone
```
git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing  
cd Silent-Face-Anti-Spoofing
```  
### Data Preprocessing
1.The training set is divided into three categories, and the pictures of the same category are put into a folder;  
2.Due to the multi-scale model fusion method, the original image and different patch are used to train the model, so the data is divided into the original map and the patch based on the Original picture;  
- Original picture(org_1_height**x**width),resize the original image to a fixed size (width, height), as shown in Figure 1;  
- Patch based on original(scale_height**x**width),The face detector is used to obtain the face frame, and the edge of the face frame is expanded according to a certain scale. In order to ensure the consistency of the input size of the model, the area of the face frame is resized to a fixed size (width, height). Fig. 2-4 shows the patch examples with scales of 1, 2.7 and 4;  
![patch demo](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/images/patch_demo.png)  

3.The Fourier spectrum is used as the auxiliary supervision, and the corresponding Fourier spectrum is generated online from the training set images.  
**The directory structure of the dataset is shown below**
```
├── datasets
    └── RGB_Images
        ├── org_1_80x60
            ├── 0
		├── aaa.png
		├── bbb.png
		└── ...
            ├── 1
		├── ddd.png
		├── eee.png
		└── ...
            └── 2
		├── ggg.png
		├── hhh.png
		└── ...
        ├── 1_80x80
        └── ...
``` 

## Generate Cropped Dataset

`generate_dataset.py` is a script designed to process images in a specified directory, crop them according to given dimensions, and save the cropped images in a structured dataset format.

### File Structure

The script processes images and organizes them into a dataset with the following structure:
```
raw_dataset
├── 0
│ ├── aaa.png
│ ├── bbb.png
│ └── ...
├── 1
│ ├── ddd.png
│ ├── eee.png
│ └── ...
└── 2
├── ggg.png
├── hhh.png
└── ...
```
In this structure, each subfolder (0, 1, 2, ...) represents a different category or class, and contains the corresponding cropped images.

### Usage

To use `generate_dataset.py`, you need to provide several command line arguments:

- `--input_dir`: The directory containing the images to be processed.
- `--save_dir`: The directory where the cropped images will be saved.
- `--device_id`: (Optional) The GPU device ID if using GPU acceleration.
- `--h_input`: (Optional) Height of the cropped image. Default is 80.
- `--w_input`: (Optional) Width of the cropped image. Default is 80.
- `--scale`: (Optional) Scale factor for bounding box adjustment.

#### Running the Script

Navigate to the directory containing `generate_dataset.py` and run the following command in your terminal:

```bash
python3 generate_dataset.py --input_dir <path_to_input_directory> --save_dir <path_to_save_directory> [--device_id <device_id>] [--h_input <height>] [--w_input <width>] [--scale <scale_factor>]
```

Replace `path_to_input_directory`, `path_to_save_directory`, `device_id`, `height`, `width`, and `scale_factor` with your desired values.

#### Example 

```bash
python3 generate_dataset.py --input_dir "./images" --save_dir "./dataset" --device_id 0 --h_input 80 --w_input 80 --scale 2.7
```

This example will process images from the `./images` directory, crop them with the specified dimensions, and save them in the `./dataset directory`, using device ID `0` and a scale factor of `2.7` for bounding box adjustment.

 
## Silence-FAS Training Script

The `train.py` script is designed for training models in the Silence-FAS project. It allows for customization of various parameters including GPU device selection, patch information, and the use of pretrained models.

### Script Parameters

The script accepts several command-line arguments to customize the training process:

- `--device_ids`: Specifies the GPU device IDs to use for training. For example, "0,1,2,3" for using four GPUs.
- `--patch_info`: Defines the patch information for training. Supported values include "org_1_80x60", "1_80x80", "2.7_80x80", "4_80x80".
- `--pretrained_model_path`: Path to the pretrained model file if available. This is optional and can be left as `None` for training from scratch.

### Usage

To use `train.py`, navigate to the directory containing the script and run the following command in your terminal:

```bash
python3 train.py --device_ids <device_ids> --patch_info <patch_info> [--pretrained_model_path <path_to_pretrained_model>]
```
Replace `device_ids`, `patch_info`, and `path_to_pretrained_model` with your desired values.

### Example
1. **For Training from Scratch:** If you want to train a model from scratch (i.e., without using a pre-trained model), you simply omit the `--pretrained_model_path` argument or set it to None. This will initialize the model with random weights and the training process will start learning the features and weights entirely from your provided training data.

	Example command for training from scratch:
	```bash
	python3 train.py --device_ids "0" --patch_info "1_80x80"
	```
2. **For Fine-Tuning:** If you want to fine-tune a model using a pre-trained model as a starting point, you should specify the path to the pre-trained model using the --pretrained_model_path argument. This allows the training process to start from the learned weights of an existing model, which can often lead to faster convergence and potentially better performance, especially when you have limited training data or are working on a similar task to the one the pre-trained model was originally used for.

	Example command for fine-tuning:
	```bash
	python3 train.py --device_ids "0" --patch_info "1_80x80" --pretrained_model_path "/path/to/pretrained/model.pth"
	```
This example will initiate training using GPUs with IDs `0`, with patch information set to `1_80x80`, and using a pretrained model located at `/path/to/pretrained/model.pth`.

In both cases, ensure the rest of the parameters (--device_ids, --patch_info, etc.) are set according to your training setup and data requirements.

### Test
 ./resources/anti_spoof_models Fusion model of in living detection  
 ./resources/detection_model Detector  
 ./images/sample Test Images  
 ```
 python test.py --image_name your_image_name
 ```    
## Reference 
- Detector [RetinaFace](https://github.com/deepinsight/insightface/tree/master/RetinaFace)  

For this project, in order to facilitate the technical exchange of developers, we created QQ group: 1121178835, welcome to join.  

In addition to the open-source silent living detection algorithm, Minivision technology also has a number of self-developed algorithms and SDK related to face recognition and human body recognition. Interested individual developers or enterprise developers can visit our website: [Mini-AI Open Platform](https://ai.minivision.cn/)
Welcome to contact us.
