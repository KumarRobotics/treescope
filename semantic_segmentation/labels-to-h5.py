import numpy as np
import open3d as o3d
import os
import argparse
import h5py

# Usage: python labels-to-h5.py -D <path-to-data>

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data", "-D", required=True, help="The main dataset directory where the labeled data is stored with the final '/' included")
parser.add_argument('--visualize', action='store_true', help='Visualize the resulting point cloud')
args = parser.parse_args()

dataset = args.data  
visualize = args.visualize  

# Create the semantic_labels directory if it doesn't exist
dataset = dataset.rstrip('/') + '/'
h5_name = dataset.rstrip('/').split('/')[-1]
semantic_dir = dataset + "semantic_labels/"
if not os.path.exists(semantic_dir):
    os.makedirs(semantic_dir)

# Check if the HDF5 file exists in the dataset folder
hdf5_output_path = os.path.join(dataset, f"{h5_name}.h5")
f = h5py.File(hdf5_output_path, 'a')
        
# Print all groups and datasets within the HDF5 file
def print_hdf5_objects(name):
    print(name)
    
f.visit(print_hdf5_objects)

# Get all the time strings from the directory
time_strings = [fname.split('.tiff')[0] for fname in os.listdir(dataset + 'scans/') if fname.endswith('.tiff')]

for time_str in time_strings:
    labels = np.load(dataset + 'converted_labels/label_' + time_str + '.npy')
    pcd = o3d.io.read_point_cloud(dataset + 'converted_scans/point_cloud_' + time_str + '.pcd')

    # Create a boolean mask for tree and ground points
    tree_mask = (labels == 8)
    ground_mask = (labels == 1)
    tree_pcd = pcd.select_by_index(np.where(tree_mask)[0])
    ground_pcd = pcd.select_by_index(np.where(ground_mask)[0])
    if visualize:
        o3d.visualization.draw_geometries([tree_pcd])
        o3d.visualization.draw_geometries([ground_pcd])

    # Place the dataset in the group
    tree_pts = np.asarray(tree_pcd.points)
    ground_pts = np.asarray(ground_pcd.points) 
    time_group = f.create_group(time_str)
    time_group.create_dataset('tree_trunks_'+time_str, data=tree_pts)
    time_group.create_dataset('ground_'+time_str, data=ground_pts)
