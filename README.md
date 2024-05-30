1. Introduction
Project Overview: The objective of this project is to develop an object detection system using YOLOv8 for identifying persons and various personal protective equipment (PPE) items from images. This involves converting annotation formats, training models, and performing inference on new images.
Dataset Description: The dataset consists of images and annotations for the following classes: person, hard-hat, gloves, mask, glasses, boots, vest, ppe-suit, ear-protector, and safety-harness. Annotations are provided in PascalVOC format and need to be converted to YOLOv8 format for training.

2. Annotation Conversion
Objective: Convert annotations from PascalVOC format to YOLOv8 format to prepare the dataset for training.
Methodology: The script pascalVOC_to_yolo.py is developed to convert the annotations. It takes three arguments: the input directory path containing PascalVOC annotations, the output directory path for YOLOv8 annotations, and the classes file.
PascalVOC to YOLO Conversion Script (pascalVOC_to_yolo.py):
Example Command:
python pascalVOC_to_yolo.py input_dir output_dir classes_file

3. Dataset Splitting
Objective: Split the dataset into training, validation, and test sets for model training and evaluation.
Methodology: The script Split_dataset.py is used to split the dataset into train, validation, and test sets. The script ensures that the ratios for each split sum to 1.
Split Dataset Script (Split_dataset.py):
Example Command:
python Split_dataset.py dataset_dir output_dir --train_ratio 0.7 --val_ratio 0.2 --test_ratio 0.1

4. Cropping and Converting Annotations
Objective: Crop images containing multiple persons and convert annotations to YOLO format for PPE detection.
Methodology: The script crop.py is used to crop images of persons and convert annotations for PPE detection.

Command:

python crop.py input_dir output_dir classes_file



5. Model Training
Objective: Train YOLOv8 models for detecting persons and PPE items.
Dataset Preparation: The dataset was split into training and validation sets. Data augmentation techniques such as flipping, scaling, and rotation were applied to enhance the model’s robustness.
Model Training: The YOLOv8 model was trained on whole images for person detection and on cropped images for PPE detection. Hyperparameters were tuned to optimize performance.
Person Detection Model:
yolo mode=train model=yolov8n.pt data=path/to/person_data.yaml epochs=50 imgsz=640
PPE Detection Model:

yolo mode=train model=yolov8n.pt data=path/to/ppe_data.yaml epochs=50 imgsz=640




6. Inference Process
Objective: Perform inference using the trained models and draw bounding boxes on detected objects.
Methodology: The inference.py script performs inference on images using the trained models. OpenCV is used to draw bounding boxes and display confidence scores.
Inference Script (inference.py):
Example Command:
python inference.py input_dir output_dir person_model ppe_model
