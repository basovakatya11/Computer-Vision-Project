# People+Mask Detection and Distance Estimation System / Project

## What is it?

The project implements a real-time people and mask detection system that combines **YOLOv8** for person detection with a **fine-tuned ResNet50** for mask classification.

The system:

1. detects people
2. identifies whether people are wearing masks or not
3. estimates distance between them

It outputs annotated videos and can provide evaluation metrics such as precision, recall, F1-score, and confusion matrices for both detection and classification performance.

The solution provides a solid foundation, though further improvements are needed.

## About ResNet50 training

The ResNet50 model, pre-trained on ImageNet, was fine-tuned for mask classification (Mask vs No_Mask) using the Face Mask Detection dataset by ashishjangra27 on Kaggle, which contains approximately 12,000 images.
[Dataset Link](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset)

Within project, the dataset is organized into directories: 'datasets/train', 'datasets/val', 'datasets/test'.

The training code is located in the file resnet50_mask_train.py.
The resulting best model weights are stored in the file resnet50_mask_best.pth and are used for inference during system execution.

## Installation

Warning: the repo takes a long time to download - contains large 'datasets' folder(13,000+ files (758 MB) tracked via Git LFS) and heavy model weights file(resnet50_mask_best.pth is ~90 MB tracked with LFS).

To set up the application, please follow these steps:

1. Clone or download this repository

2. Navigate to the project directory.

3. It’s always best practice to work within a virtual environment to prevent conflicts with your system-wide Python packages.
   The set of commands will be (you should have _venv_ module installed on your machine to create and manage virtual environments):

on Windows:

```shell
python -m venv .venv
.venv\Scripts\activate
```

on macOS/Linux:

```shell
python -m venv .venv
source .venv/bin/activate
```

4. Ensure that pip is correctly installed on your machine. Then, install the required dependencies by running the following command:

```shell
pip install -r requirements.txt
```

## Usage

To run the video processing function, execute the following command in your terminal:

```shell
python main.py
```

By default, the script will take _input.mp4_ as input and save the annotated result as _output_demo.mp4_.

### Evaluation

(Optional)

- ResNet50 performance (on the test part of the main dataset - directory 'datasets/test')

To evaluate the fine-tuned ResNet50 on the test portion of the dataset it was trained with, run:

```shell
python test_resnet50.py
```

The script will give you the confusion matrix plot for ResNet50 and the precision, recall, and F1-score metrics displayed in the terminal

- Overall system performance (on the complex annotated dataset - directory 'datasets/final')

To evaluate the overall system performance, uncomment the validate_system function at the end of main.py and comment out the process_video block (to prevent unintended executions). Then run the following command in your terminal:

```shell
python main.py
```

The script evaluates the overall system on the annotated dataset. For this project evaluation, we used the Face Mask Detection dataset by andrewmvd on Kaggle ([Dataset Link](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)), which contains labeled images of people with and without masks. The 'incorrect masks' category was omitted for the purposes of this study.

Upon execution, the script generates confusion matrix plots for both YOLOv8 detection performance and the overall system performance, along with the corresponding precision, recall, and F1-score metrics displayed in the terminal.

## Termination

(When you're done working with the project, you can deactivate the virtual environment):

```shell
deactivate
```
