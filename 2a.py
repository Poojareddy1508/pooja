import os as os
import xml.etree.ElementTree as ET
from PIL import Image
from pathlib import Path

# Define the paths to your dataset
input_dir = r"C:\Users\anush\OneDrive\Pictures\dingo"
output_dir = r"C:\Users\anush\OneDrive\Pictures\dingo\crop"

# Define the class names and corresponding directories
class_names = [
    "Staffordshire_bullterrier im",
    "newfoundland im",
    "Pembroke im",
    "dingo im"
]

# Define a function to get bounding box information from the annotations dataset
def get_bounding_boxes(annot):
    xml = annot
    tree = ET.parse(xml)
    root = tree.getroot()
    objects = root.findall('object')
    bbox = []
    for o in objects:
        bndbox = o.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        bbox.append((xmin, ymin, xmax, ymax))
    return bbox

# Iterate over class directories
for class_name in class_names:
    # Define paths to images and annotations for the current class
    images_dir = os.path.join(input_dir, "Images", class_name)
    annotations_dir = os.path.join(input_dir, "Annotations", class_name)
    
    # Get a list of image file paths for the current class
    image_files = [os.path.join(images_dir, file) for file in os.listdir(images_dir) if file.endswith('.jpg')]
    
    # Create a subdirectory within the output directory for the current class
    class_output_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_output_dir, exist_ok=True)
    
    # Iterate over image files
    for image_file in image_files:
        # Get bounding box information for the current image
        annotation_file = os.path.join(annotations_dir, os.path.splitext(os.path.basename(image_file))[0])  # Assuming annotation files have no extensions
        bbox = get_bounding_boxes(annotation_file)
        
        # Load the image
        img = Image.open(image_file)
        
        for j in range(len(bbox)):
            # Crop the image using the bounding box
            x_min, y_min, x_max, y_max = bbox[j]
            cropped_img = img.crop((x_min, y_min, x_max, y_max))
            
            # Resize the cropped image to 100x100 pixels
            cropped_img = cropped_img.resize((100, 100))
            
            # Define the new file path for the cropped and resized image within the class subdirectory
            new_path = os.path.join(class_output_dir, os.path.basename(image_file).replace('.jpg', f'-{j}.jpg'))
            
            # Save the cropped and resized image
            cropped_img = cropped_img.convert('RGB')
            cropped_img.save(new_path)
