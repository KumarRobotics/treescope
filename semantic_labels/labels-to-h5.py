import numpy as np
import open3d as o3d
import os
import argparse
import h5py

# Usage: python labels-to-h5.py --dataset <dataset-name>
# Example: python labels-to-h5.py  --dataset UCM-0822

# Default values
visualize = False
dataset_path = '/derek_data/dataset/labels/'
dataset = 'UCM-0523-M'

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='', help='Dataset directory')
parser.add_argument('--time', type=str, default='', help='Time string')
parser.add_argument('--visualize', action='store_true', help='Visualize the resulting point cloud')
args = parser.parse_args()

if args.dataset:
    dataset = args.dataset
if args.visualize:
    visualize = True

# Add trailing backslash to dataset directory if missing
dataset_path = dataset_path + dataset
dataset_path = dataset_path.rstrip('/') + '/'

# Create the semantic_labels directory if it doesn't exist
semantic_dir = dataset_path + "semantic_labels/"
if not os.path.exists(semantic_dir):
    os.makedirs(semantic_dir)

# Check if the HDF5 file exists in the dataset_path folder
# hdf5_output_path = os.path.join("h5_labels", dataset + '.h5')
hdf5_output_path = dataset + '.h5'

# Use mode 'a' to append to the existing file if it exists, otherwise create a new file
f = h5py.File(hdf5_output_path, 'a')
        
# Print all groups and datasets within the HDF5 file
def print_hdf5_objects(name):
    print(name)
    
f.visit(print_hdf5_objects)

# Get all the time strings from the directory
time_strings = [fname.split('.tiff')[0] for fname in os.listdir(dataset_path + 'scans/') if fname.endswith('.tiff')]

for time_str in time_strings:
    labels = np.load(dataset_path + 'converted_labels/label_' + time_str + '.npy')

    # Load the point cloud from the PCD file
    pcd = o3d.io.read_point_cloud(dataset_path + 'converted_scans/point_cloud_' + time_str + '.pcd')

    # Create a boolean mask for tree and ground points
    tree_mask = (labels == 8)
    ground_mask = (labels == 1)

    # Apply the mask to the point cloud
    tree_pcd = pcd.select_by_index(np.where(tree_mask)[0])
    ground_pcd = pcd.select_by_index(np.where(ground_mask)[0])

    # Visualize the resulting point cloud
    if visualize:
        o3d.visualization.draw_geometries([tree_pcd])
        o3d.visualization.draw_geometries([ground_pcd])

    # Place the dataset in the group for this time string
    tree_pts = np.asarray(tree_pcd.points)
    ground_pts = np.asarray(ground_pcd.points)  # Corrected this line
    time_group = f.create_group(time_str)
    time_group.create_dataset('tree_trunks_'+time_str, data=tree_pts)
    time_group.create_dataset('ground_'+time_str, data=ground_pts)
