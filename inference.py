import argparse  # Import argparse for command-line argument parsing
import cv2  # Import OpenCV for image processing
import os  # Import os for file and directory operations
from ultralytics import YOLO  # Import YOLO from ultralytics library

# Define class names for PPE items
PPE_CLASS_NAMES = {
    0: 'Hard-hat',
    1: 'Gloves',
    2: 'Mask',
    3: 'Glasses',
    4: 'Boots',
    5: 'Vest',
    6: 'PPE-suit',
}

# Function to perform inference on images
def perform_inference(input_dir, output_dir, person_model_path, ppe_model_path):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create output directory if it does not exist
    
    # Load the YOLO models for person and PPE detection
    person_model = YOLO(person_model_path)
    ppe_model = YOLO(ppe_model_path)
    
    # Iterate over each file in the input directory
    for file in os.listdir(input_dir):
        if file.endswith('.jpg') or file.endswith('.png'):
            image_path = os.path.join(input_dir, file)
            image = cv2.imread(image_path)  # Read the image using OpenCV
            h, w = image.shape[:2]  # Get the height and width of the image
            
            # Perform person detection
            person_results = person_model.predict(image)
            persons = person_results[0].boxes.xyxy.numpy()  # Get person bounding boxes
            person_confidences = person_results[0].boxes.conf.numpy()  # Get confidence scores for persons
            
            for i, (x1, y1, x2, y2) in enumerate(persons):
                # Ensure coordinates are within image boundaries
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                
                # Draw person bounding box in green
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f'Person ({person_confidences[i]:.2f})'
                cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Crop the person region from the image
                cropped_img = image[int(y1):int(y2), int(x1):int(x2)]
                ch, cw = cropped_img.shape[:2]
                
                # Perform PPE detection on the cropped person region
                ppe_results = ppe_model.predict(cropped_img)
                ppe_items = ppe_results[0].boxes.xyxy.numpy()  # Get PPE bounding boxes
                ppe_labels = ppe_results[0].boxes.cls.numpy()  # Get class labels for PPE
                ppe_confidences = ppe_results[0].boxes.conf.numpy()  # Get confidence scores for PPE
                
                for j, (px1, py1, px2, py2) in enumerate(ppe_items):
                    # Transform PPE coordinates back to original image
                    px1, py1, px2, py2 = int(px1 + x1), int(py1 + y1), int(px2 + x1), int(py2 + y1)
                    px1, py1, px2, py2 = max(0, px1), max(0, py1), min(w, px2), min(h, py2)
                    
                    # Draw PPE bounding box in red
                    cv2.rectangle(image, (px1, py1), (px2, py2), (255, 0, 0), 2)
                    class_name = PPE_CLASS_NAMES.get(int(ppe_labels[j]), 'Unknown')
                    ppe_label = f'{class_name} ({ppe_confidences[j]:.2f})'
                    cv2.putText(image, ppe_label, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Save the image with bounding boxes to the output directory
            cv2.imwrite(os.path.join(output_dir, file), image)

# Main function to parse command-line arguments and call the perform_inference function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory with input images")
    parser.add_argument("output_dir", help="Directory to save output images")
    parser.add_argument("person_model", help="Path to the person detection model")
    parser.add_argument("ppe_model", help="Path to the PPE detection model")
    args = parser.parse_args()

    perform_inference(args.input_dir, args.output_dir, args.person_model, args.ppe_model)
