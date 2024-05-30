import os  # Import os for file and directory operations
import cv2  # Import OpenCV for image processing
import argparse  # Import argparse for command-line argument parsing
import xml.etree.ElementTree as ET  # Import ElementTree for parsing XML files

# Function to crop images containing multiple persons and convert annotations to YOLO format
def crop_multiple_persons_and_convert_annotations(input_dir, output_dir, classes_file):
    # Read the classes from the classes_file
    with open(classes_file, 'r') as f:
        classes = f.read().strip().split('\n')
    
    # Get the index of the 'person' class
    person_class_id = classes.index('person') if 'person' in classes else -1
    if person_class_id == -1:
        raise ValueError("Person class not found in classes file")

    # Set up directories for annotations, images, cropped images, and YOLO annotations
    annotations_dir = os.path.join(input_dir, 'annotations')
    images_dir = os.path.join(input_dir, 'images')
    cropped_images_dir = os.path.join(output_dir, 'images')
    yolo_annotations_dir = os.path.join(output_dir, 'labels')
    
    os.makedirs(cropped_images_dir, exist_ok=True)
    os.makedirs(yolo_annotations_dir, exist_ok=True)
    
    print(f"Cropped images will be saved to: {cropped_images_dir}")
    print(f"YOLO annotations will be saved to: {yolo_annotations_dir}")
    
    # Iterate over each annotation file in the annotations directory
    for annotation_file in os.listdir(annotations_dir):
        if not annotation_file.endswith('.xml'):
            continue
        
        image_file = os.path.splitext(annotation_file)[0] + '.jpg'
        image_path = os.path.join(images_dir, image_file)
        
        # Check if the corresponding image file exists
        if not os.path.exists(image_path):
            print(f"Warning: Image file {image_file} not found. Skipping annotation file {annotation_file}.")
            continue
        
        # Parse the XML annotation file
        tree = ET.parse(os.path.join(annotations_dir, annotation_file))
        root = tree.getroot()
        
        # Read the image using OpenCV
        image = cv2.imread(image_path)
        image_height, image_width = image.shape[:2]
        
        person_bboxes = []
        # Collect bounding boxes for person objects
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name == 'person':
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                person_bboxes.append((xmin, ymin, xmax, ymax))
        
        # Iterate over each person bounding box and process
        for i, (xmin, ymin, xmax, ymax) in enumerate(person_bboxes):
            cropped_image = image[ymin:ymax, xmin:xmax]
            cropped_image_file = f"{os.path.splitext(image_file)[0]}_person_{i}.jpg"
            cropped_image_path = os.path.join(cropped_images_dir, cropped_image_file)
            cv2.imwrite(cropped_image_path, cropped_image)
            
            yolo_annotations = []
            
            # Collect bounding boxes for non-person objects and convert to YOLO format
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name == 'person':
                    continue
                
                class_id = classes.index(class_name) if class_name in classes else -1
                if class_id == -1:
                    continue
                
                # Adjust class_id to exclude 'person'
                class_id -= 1
                
                bbox = obj.find('bndbox')
                obj_xmin = int(bbox.find('xmin').text)
                obj_ymin = int(bbox.find('ymin').text)
                obj_xmax = int(bbox.find('xmax').text)
                obj_ymax = int(bbox.find('ymax').text)
                
                # Adjust coordinates relative to the cropped image
                new_xmin = max(0, obj_xmin - xmin)
                new_ymin = max(0, obj_ymin - ymin)
                new_xmax = min(xmax - xmin, obj_xmax - xmin)
                new_ymax = min(ymax - ymin, obj_ymax - ymin)

                # Ensure new coordinates are within bounds
                new_xmin = max(0, min(new_xmin, xmax - xmin))
                new_ymin = max(0, min(new_ymin, ymax - ymin))
                new_xmax = max(0, min(new_xmax, xmax - xmin))
                new_ymax = max(0, min(new_ymax, ymax - ymin))
                
                if new_xmax <= new_xmin or new_ymax <= new_ymin:
                    continue
                
                center_x = (new_xmin + new_xmax) / 2.0 / (xmax - xmin)
                center_y = (new_ymin + new_ymax) / 2.0 / (ymax - ymin)
                bbox_width = (new_xmax - new_xmin) / float(xmax - xmin)
                bbox_height = (new_ymax - new_ymin) / float(ymax - ymin)
                
                # Ensure coordinates are normalized and within bounds
                if 0 <= center_x <= 1 and 0 <= center_y <= 1 and 0 <= bbox_width <= 1 and 0 <= bbox_height <= 1:
                    annotation = f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}"
                    yolo_annotations.append(annotation)
            
            # Write YOLO annotations to a file if there are any
            if yolo_annotations:
                yolo_annotation_file = f"{os.path.splitext(image_file)[0]}_person_{i}.txt"
                yolo_annotation_path = os.path.join(yolo_annotations_dir, yolo_annotation_file)
                with open(yolo_annotation_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
    
    print("Cropped images and YOLO annotations prepared.")

# Main function to parse command-line arguments and call the crop_multiple_persons_and_convert_annotations function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop multiple person images and prepare YOLO annotations for PPE detection")
    parser.add_argument('input_dir', type=str, help="Path to the base input directory")
    parser.add_argument('output_dir', type=str, help="Path to the output directory for cropped images and YOLO annotations")
    parser.add_argument('classes_file', type=str, help="Path to the classes.txt file")

    args = parser.parse_args()
    crop_multiple_persons_and_convert_annotations(args.input_dir, args.output_dir, args.classes_file)
