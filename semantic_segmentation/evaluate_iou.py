"""
python evaluate_iou.py ground_truth.png prediction.png wharton-forest.yaml
unlabelled IoU: 0.9333916582292169
road IoU: 0.8720882078890154
tree_trunk IoU: 0.7105739475681673
Mean IoU: 0.8386846045621331

python evaluate_iou.py stacked_image.png  wharton-forest.yaml 
unlabelled IoU: 0.9333916582292169
road IoU: 0.8720882078890154
tree_trunk IoU: 0.7105739475681673
Mean IoU: 0.8386846045621331
"""


import cv2
import numpy as np
import yaml
import argparse

def create_mask_for_color(image, color):
    """Create a binary mask where the given color is white and all other colors are black."""
    color_mask = np.all(image == color, axis=-1)
    return color_mask.astype(np.uint8)

def calculate_iou(ground_truth_mask, prediction_mask):
    """Calculate the Intersection over Union (IoU) for a single class."""
    intersection = np.logical_and(ground_truth_mask, prediction_mask)
    union = np.logical_or(ground_truth_mask, prediction_mask)
    if np.sum(union) == 0:
        return float('nan') 
    else:
        return np.sum(intersection) / np.sum(union)

def read_yaml(file_path):
    """Read a YAML file and return the contents."""
    with open(file_path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    return data_loaded

def load_images(range_image=None, predicted_image=None, labeled_image=None):
    gt_image, pred_image = None, None

    # If a single range image is provided, split stacked image
    if range_image is not None:
        full_image = cv2.imread(range_image)
        if full_image is None:
            raise ValueError("Range image not found or the path is incorrect")

        image_height = full_image.shape[0]
        segment_height = image_height // 3
        range_img = full_image[:segment_height, :]
        gt_image = full_image[segment_height:2*segment_height, :]
        pred_image = full_image[2*segment_height:3*segment_height, :]

        # Save the segments to disk
        base_name = range_image.rsplit('.', 1)[0] 
        cv2.imwrite(f'{base_name}-range_image.png', range_img)
        cv2.imwrite(f'{base_name}-ground_truth.png', gt_image)
        cv2.imwrite(f'{base_name}-pred.png', pred_image)


    # If separate images for prediction and labeled ground truth are provided, load them.
    elif predicted_image is not None and labeled_image is not None:
        gt_image = cv2.imread(labeled_image)
        pred_image = cv2.imread(predicted_image)

        if gt_image is None or pred_image is None:
            raise ValueError("One of the images is not found or the paths are incorrect")

    else:
        raise ValueError("Incorrect image arguments provided")

    return gt_image, pred_image

        
def main(args):
    if len(args.images) == 2:
        # If there are only two arguments, the first is the stacked image, and the second is the YAML file.
        range_image = args.images[0]
        yaml_path = args.images[1]
        ground_truth_image, predicted_image = load_images(range_image=range_image)
    elif len(args.images) == 3:
        # If there are three arguments, the first two are separate images, and the third is the YAML file.
        ground_truth_image_path, prediction_image_path, yaml_path = args.images
        ground_truth_image, predicted_image = load_images(predicted_image=prediction_image_path, labeled_image=ground_truth_image_path)
    else:
        raise ValueError("Invalid number of arguments provided.")
    
    # Read the YAML file for color codes
    yaml_data = read_yaml(yaml_path)
    color_codes = {int(k): tuple(v) for k, v in yaml_data['color_map'].items()}
    classes = yaml_data['labels']
    ious = []
    # Calculate IoU for each class
    for class_id, color in color_codes.items():
        ground_truth_mask = create_mask_for_color(ground_truth_image, color)
        prediction_mask = create_mask_for_color(predicted_image, color)
        iou = calculate_iou(ground_truth_mask, prediction_mask)

        if not np.isnan(iou):
            ious.append(iou)
            class_name = classes[class_id]
            print(f"{class_name} IoU: {iou}")

    # Calculate the mean IoU, ignoring NaN values
    mean_iou = np.nanmean(ious)
    print(f"Mean IoU: {mean_iou}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate IoU for segmentation maps.')
    parser.add_argument('images', type=str, nargs='+', help='Path to the image file(s) followed by the YAML file.')
    args = parser.parse_args()

    main(args)