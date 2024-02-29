import numpy as np
import open3d as o3d
import h5py
import os
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Unpack HDF5 files into NPY and PCD files.")
parser.add_argument('--file', type=str, required=True, help='Path to the input HDF5 file.')
parser.add_argument('--output', type=str, required=True, help='Directory to save the output files.')
parser.add_argument('--visualize', action='store_true', help='Visualize the resulting point clouds.')
args = parser.parse_args()

def unpack_hdf5(hdf5_path, output_path, visualize=False):
    with h5py.File(hdf5_path, 'r') as f:
        for time_str in f.keys():
            group = f[time_str]
            tree_pts = group['tree_trunks_' + time_str][:]
            ground_pts = group['ground_' + time_str][:]

            # labels: trees as 8, ground as 1
            labels = np.zeros((tree_pts.shape[0] + ground_pts.shape[0],), dtype=int)
            labels[:tree_pts.shape[0]] = 8  # Label for trees
            labels[tree_pts.shape[0]:] = 1  # Label for ground

            # Save labels as .npy
            npy_path = os.path.join(output_path, 'converted_labels', 'label_' + time_str + '.npy')
            os.makedirs(os.path.dirname(npy_path), exist_ok=True)
            np.save(npy_path, labels)
            
            # Merge tree and ground points for the point cloud
            all_pts = np.vstack((tree_pts, ground_pts))
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(all_pts)
            
            # Save point cloud as .pcd
            pcd_path = os.path.join(output_path, 'converted_scans', 'point_cloud_' + time_str + '.pcd')
            os.makedirs(os.path.dirname(pcd_path), exist_ok=True)
            o3d.io.write_point_cloud(pcd_path, pcd)

            # Optionally visualize the point clouds
            if visualize:
                o3d.visualization.draw_geometries([pcd])

unpack_hdf5(args.file, args.output, args.visualize)
