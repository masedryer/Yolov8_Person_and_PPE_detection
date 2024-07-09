import argparse
import cv2
import os
from ultralytics import YOLO

PPE_CLASS_NAMES = {
    0: 'Hard-hat',
    1: 'Gloves',
    2: 'Mask',
    3: 'Glasses',
    4: 'Boots',
    5: 'Vest',
    6: 'PPE-suit',
}

REQUIRED_PPE_CLASSES = {0, 1, 4, 5}  # Hard-hat, Gloves, Boots, Vest

def perform_inference_single_image(image_path, output_path, person_model_path, ppe_model_path):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))  # Create output directory if it does not exist
    
    person_model = YOLO(person_model_path)
    ppe_model = YOLO(ppe_model_path)
    
    image = cv2.imread(image_path)  # Read the image using OpenCV
    h, w = image.shape[:2]  # Get the height and width of the image
    
    # Perform person detection
    person_results = person_model.predict(image)
    persons = person_results[0].boxes.xyxy.numpy()  # Get person bounding boxes
    person_confidences = person_results[0].boxes.conf.numpy()  # Get confidence scores for persons
    
    for i, (x1, y1, x2, y2) in enumerate(persons):
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        
        # Crop the person region from the image
        cropped_img = image[int(y1):int(y2), int(x1):int(x2)]
        ch, cw = cropped_img.shape[:2]
        
        # Perform PPE detection on the cropped person region
        ppe_results = ppe_model.predict(cropped_img)
        ppe_items = ppe_results[0].boxes.xyxy.numpy()  # Get PPE bounding boxes
        ppe_labels = ppe_results[0].boxes.cls.numpy()  # Get class labels for PPE
        ppe_confidences = ppe_results[0].boxes.conf.numpy()  # Get confidence scores for PPE
        
        detected_ppe_classes = set()
        
        for j, (px1, py1, px2, py2) in enumerate(ppe_items):
            px1, py1, px2, py2 = int(px1 + x1), int(py1 + y1), int(px2 + x1), int(py2 + y1)
            px1, py1, px2, py2 = max(0, px1), max(0, py1), min(w, px2), min(h, py2)
            
            detected_ppe_classes.add(int(ppe_labels[j]))
            
            # Draw PPE bounding box in red
            cv2.rectangle(image, (px1, py1), (px2, py2), (255, 0, 0), 2)
            class_name = PPE_CLASS_NAMES.get(int(ppe_labels[j]), 'Unknown')
            ppe_label = f'{class_name} ({ppe_confidences[j]:.2f})'
            cv2.putText(image, ppe_label, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Determine the color of the person bounding box
        if REQUIRED_PPE_CLASSES.issubset(detected_ppe_classes):
            person_color = (0, 255, 0)  # Green if all required PPE are detected
        else:
            person_color = (0, 0, 255)  # Red otherwise
        
        # Draw person bounding box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), person_color, 2)
        label = f'Person ({person_confidences[i]:.2f})'
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, person_color, 2)
    
    cv2.imwrite(output_path, image)  # Save the image with bounding boxes to the output directory

if __name__ == "__main__":
    # Default paths
    default_image_path = "D:\LTA\AI Model\PPE Detection\Yolov8_Person_and_PPE_detection\input\image3.JPG"
    default_output_path = "D:\LTA\AI Model\PPE Detection\Yolov8_Person_and_PPE_detection\output\output3.jpg"
    default_person_model_path = "D:\LTA\AI Model\PPE Detection\Yolov8_Person_and_PPE_detection\weights\person.pt"
    default_ppe_model_path = "D:\LTA\AI Model\PPE Detection\Yolov8_Person_and_PPE_detection\weights\ppe.pt"
    
    # Call the inference function with default paths
    perform_inference_single_image(default_image_path, default_output_path, default_person_model_path, default_ppe_model_path)
