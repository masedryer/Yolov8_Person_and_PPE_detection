import os  # Import the os module for file and directory operations
import xml.etree.ElementTree as ET  # Import ElementTree for parsing XML files
import argparse  # Import argparse for command-line argument parsing

# Function to filter annotations and convert them to YOLO format
def filter_person_annotations(input_dir, output_dir, classes_file):
    # Read the classes from the classes_file
    with open(classes_file, 'r') as f:
        classes = f.read().strip().split('\n')
    
    # Get the index of the 'person' class
    person_class_id = classes.index('person') if 'person' in classes else -1
    if person_class_id == -1:
        raise ValueError("Person class not found in classes file")

    # Set up the directories for annotations and images
    annotations_dir = os.path.join(input_dir, 'annotations')
    images_dir = os.path.join(input_dir, 'images')
    
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Iterate through each annotation file in the annotations directory
    for annotation_file in os.listdir(annotations_dir):
        if not annotation_file.endswith('.xml'):
            continue
        
        # Parse the XML annotation file
        tree = ET.parse(os.path.join(annotations_dir, annotation_file))
        root = tree.getroot()
        
        # Get the dimensions of the image
        image_width = int(root.find('size/width').text)
        image_height = int(root.find('size/height').text)
        
        yolo_annotations = []  # List to store YOLO annotations
        valid = True  # Flag to check if the annotation is valid
        
        # Iterate through each object in the annotation file
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name != 'person':
                continue  # Skip if the class is not 'person'
            
            class_id = person_class_id  # Set the class ID to person_class_id
            
            # Get the bounding box coordinates
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            # Convert the bounding box coordinates to YOLO format
            center_x = (xmin + xmax) / 2.0 / image_width
            center_y = (ymin + ymax) / 2.0 / image_height
            bbox_width = (xmax - xmin) / float(image_width)
            bbox_height = (ymax - ymin) / float(image_height)
            
            # Append the converted annotation to the list
            yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}")
        
        # Write the YOLO annotations to a file if there are any
        if yolo_annotations:
            output_file = os.path.join(output_dir, os.path.splitext(annotation_file)[0] + '.txt')
            with open(output_file, 'w') as f:
                f.write('\n'.join(yolo_annotations))
        else:
            # Remove the invalid annotation file and the corresponding image file
            os.remove(os.path.join(annotations_dir, annotation_file))
            image_file = os.path.splitext(annotation_file)[0] + '.jpg'
            image_path = os.path.join(images_dir, image_file)
            if os.path.exists(image_path):
                os.remove(image_path)
            else:
                print(f"Warning: Corresponding image file {image_file} not found and could not be removed.")

    # Remove any empty label files along with their images
    for label_file in os.listdir(output_dir):
        label_path = os.path.join(output_dir, label_file)
        if os.path.getsize(label_path) == 0:
            os.remove(label_path)
            image_file = os.path.splitext(label_file)[0] + '.jpg'
            image_path = os.path.join(images_dir, image_file)
            if os.path.exists(image_path):
                os.remove(image_path)
            else:
                print(f"Warning: Corresponding image file {image_file} not found and could not be removed.")

# Main function to parse command-line arguments and call the filter_person_annotations function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter annotations to include only the person class")
    parser.add_argument('input_dir', type=str, help="Path to the base input directory")
    parser.add_argument('output_dir', type=str, help="Path to the output directory for YOLOv8 annotations")
    parser.add_argument('classes_file', type=str, help="Path to the classes.txt file")

    args = parser.parse_args()
    filter_person_annotations(args.input_dir, args.output_dir, args.classes_file)
