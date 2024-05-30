import os  # Import the os module for file and directory operations
import shutil  # Import shutil for copying files
import random  # Import random for shuffling the list of images
import argparse  # Import argparse for command-line argument parsing
import math  # Import math for mathematical operations

# Function to split the dataset into training, validation, and test sets
def split_dataset(dataset_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    # Calculate the total ratio to ensure it sums to 1
    total_ratio = train_ratio + val_ratio + test_ratio
    print(f"Total ratio: {total_ratio}")
    assert math.isclose(total_ratio, 1.0, rel_tol=1e-9), "The ratios must sum up to 1."

    # Set up directories for images and annotations
    images_dir = os.path.join(dataset_dir, 'images')
    annotations_dir = os.path.join(dataset_dir, 'labels')

    # List all image files in the images directory
    images = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]
    random.shuffle(images)  # Shuffle the list of images

    # Filter images to ensure each image has a corresponding annotation file
    valid_images = [img for img in images if os.path.exists(os.path.join(annotations_dir, img.replace('.jpg', '.txt').replace('.png', '.txt')))]

    # Calculate the cutoffs for splitting the dataset
    train_cutoff = int(train_ratio * len(valid_images))
    val_cutoff = int((train_ratio + val_ratio) * len(valid_images))

    # Split the images into training, validation, and test sets
    train_images = valid_images[:train_cutoff]
    val_images = valid_images[train_cutoff:val_cutoff]
    test_images = valid_images[val_cutoff:]

    # Dictionary to hold the split datasets
    splits = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }

    # Create directories for each split and copy the files
    for split, split_images in splits.items():
        split_images_dir = os.path.join(output_dir, split, 'images')
        split_annotations_dir = os.path.join(output_dir, split, 'labels')
        os.makedirs(split_images_dir, exist_ok=True)  # Create image directory for the split
        os.makedirs(split_annotations_dir, exist_ok=True)  # Create annotation directory for the split

        # Copy each image and its corresponding annotation file to the split directories
        for image in split_images:
            annotation_file = image.replace('.jpg', '.txt').replace('.png', '.txt')
            if os.path.exists(os.path.join(images_dir, image)) and os.path.exists(os.path.join(annotations_dir, annotation_file)):
                shutil.copy(os.path.join(images_dir, image), os.path.join(split_images_dir, image))
                shutil.copy(os.path.join(annotations_dir, annotation_file), os.path.join(split_annotations_dir, annotation_file))

# Main function to parse command-line arguments and call the split_dataset function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train, val, and test sets.")
    parser.add_argument("dataset_dir", type=str, help="Path to the dataset directory.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory.")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of training set.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Ratio of validation set.")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of test set.")

    args = parser.parse_args()
    
    # Debugging output for the passed arguments
    print(f"Train ratio: {args.train_ratio}, Val ratio: {args.val_ratio}, Test ratio: {args.test_ratio}")

    # Call the function to split the dataset
    split_dataset(args.dataset_dir, args.output_dir, args.train_ratio, args.val_ratio, args.test_ratio)
