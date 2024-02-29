#! /usr/bin/env python3
# title			:
# description	:
# author		:Ankit Prabhu

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import ros_numpy
import sys
import numpy as np
from sklearn.cluster import OPTICS, DBSCAN
import time
import open3d as o3d
import json
import webcolors
import matplotlib.pyplot as plt
import pandas as pd
import open3d.visualization.rendering as rendering
import csv
import os

# TODO 4) Test for other bags to see results
class ProcessTreeNode:

    def __init__(self):

        self.tree_cloud_sub = rospy.Subscriber(
            "/tree_cloud_world", PointCloud2, callback=self.tree_pc, queue_size=10)
        self.clus_tree_cloud_pub = rospy.Publisher(
            "/tree_inst", PointCloud2, queue_size=100)
        
        # Tunable Params
        # *****************************
        # Param Tuning Cases
        ####################################
        # When Trees are close together:-
        # Use conservative clustering for separation init-[0.1, 10] control-[0.15, 25]
        # The issues with that is cannot measure till long height especially for thin trees
        ######################################
        ######################################
        # When Trees are Far Away (Less Noise):-
        # init - [0.3, 20]
        # control - [(0.15,15/20),(0.17,20-> seems to work better)]
        # For Tree 3 (Very Noise) [init 0.1, 20; control 0.15,33]
        ########################################
        # For noisy tree and split branching
        # init_db-[0.1, 25/15], else default is [0.3, 75/50] (may miss top points, use for noisy spots)
        # control_db - [0.15, 20/15] else default [0.5, 50] (heuristic)
        # init_min_clus - 200 for noise 500 for clean (low value as clusters are more clean)
        # control_min_clus- 400 or 500 (heuristic)
        # **********************************

        # DBSCAN cluster params
        self.valid_range_threshold = 10000

        # to find the core point (lenient)
        self.init_min_sample_threshold_dbscan = 30 #[20]
        self.init_epsilon_value_dbscan = 0.5  #[0.3]
        self.init_min_cluster_pts = 100 # throws out an cluster size less than this (gets rid of false positives)

        # once you've found the core point (strict to throw out false positives)
        self.control_pt_min_sample_dbscan = 80 #[20]
        self.control_pt_eps_dbscan = 0.2 #[0.17]
        self.control_pt_min_cluster_pts = 1000 #400

        # post processing of json file
        self.step_size_for_control_pts = 10 #15 how well you see green and red pts
        self.kdtree_search_radius = 1
        self.slash_height_thresh = 0.03
        self.number_to_round_to = 3
        self.per_tree_profile_height = 0.5
        self.min_slash_points_for_median_diameter = 2
        self.step_size_for_tree_diameters = 0.1

        self.dbh_percentile = 50
        self.max_dbh = 0.6 # throw out values greater than this
        self.control_pts_cluster_time = 220 # cluster accumulation period

        # Flags
        self.show_slashes = False # show the red/green pts (to debug the init and control params)
        self.compute_tree_profile = True # do the post processing of DBH estimations
        self.viz_clustered_control_points = True # click in vispy to see which cluster is which id
        self.plot_tree_profile = False
        self.save_stats_as_df = False
        self.save_stats_as_csv = False
        self.converted_indices_filename = None
        self.data_path = "data/" # include the last slash
        
        # Load trajectory data from traj.txt
        self.traj_file = None

        self.pc_fields_ = make_fields()
        self.control_points_dict = {}
        self.cluster_points_dict = {}
        self.last_time_stamp = None
        # Permanently generate 500 colors for clusters
        self.perma_clus_color= self.gen_rand_colors(500)

    def tree_pc(self, msg):

        # Numpifing the point cloud msg
        cur_timestamp = msg.header.stamp
        if self.last_time_stamp == None:
            self.last_time_stamp = cur_timestamp

        tree_cloud_full = ros_numpy.numpify(msg)
        # Replacing NaNs
        x_coords = np.nan_to_num(tree_cloud_full['x'].flatten())
        y_coords = np.nan_to_num(tree_cloud_full['y'].flatten())
        z_coords = np.nan_to_num(tree_cloud_full['z'].flatten())
        # 10.0 as place holder for NaNs present in intensities
        mod_intensities = np.nan_to_num(
            tree_cloud_full['intensity'].flatten(), nan=10.0)
        tree_cloud_full = np.zeros((x_coords.shape[0], 4))
        tree_cloud_full[:, 0] = x_coords
        tree_cloud_full[:, 1] = y_coords
        tree_cloud_full[:, 2] = z_coords
        tree_cloud_full[:, 3] = mod_intensities

        # Selecting only tree points
        tree_valid_idx = tree_cloud_full[:, 3] == 8
        tree_cloud = tree_cloud_full[tree_valid_idx, :]

        # Thresholding by Range
        valid_range_idx = self.threshold_by_range(
            self.valid_range_threshold, tree_cloud) # Param (Range Thresholding)
        tree_cloud = tree_cloud[valid_range_idx, :]

        # Clustering Trees
        tree_labels = self.cluster_tree(
            pc_xyz=tree_cloud, min_samples=self.init_min_sample_threshold_dbscan, eps=self.init_epsilon_value_dbscan) # Param (DBSCAN n and eps)

        clustered_tree_cloud = np.zeros_like(tree_cloud)
        clustered_tree_cloud[:, 0] = tree_cloud[:, 0]
        clustered_tree_cloud[:, 1] = tree_cloud[:, 1]
        clustered_tree_cloud[:, 2] = tree_cloud[:, 2]
        clustered_tree_cloud[:, 3] = tree_labels
        clean_clus_idx = clustered_tree_cloud[:, 3] != -1
        clustered_tree_cloud = clustered_tree_cloud[clean_clus_idx, :]

        self.publish_clustered_cloud(
            clustered_tree_cloud, cur_timestamp, "odom", self.clus_tree_cloud_pub)

        # Working on individual tree segment
        tree_instances = np.unique(clustered_tree_cloud[:, 3])
        for count, inst_idx in enumerate(tree_instances):

            cur_instant_pc_xyz = np.around(
                clustered_tree_cloud[clustered_tree_cloud[:, 3] == inst_idx, :3], self.number_to_round_to)
            
            if cur_instant_pc_xyz.shape[0] < self.init_min_cluster_pts: # Param (Filter initial clusters to remove noisy clusters)
                continue

            o_pcd = o3d.geometry.PointCloud()
            o_pcd.points = o3d.utility.Vector3dVector(cur_instant_pc_xyz)
            o_pcd_tree = o3d.geometry.KDTreeFlann(o_pcd)

            o_pcd.paint_uniform_color([0.3, 0.3, 0.3])

            z_profile, z_profile_unique_idx = np.unique(
                cur_instant_pc_xyz[:, 2], return_index=True)
            z_profile_unique_idx_sorted = z_profile_unique_idx[np.argsort(
                z_profile)]
            
            z_profile_percentile_selected_idx = z_profile_unique_idx_sorted[np.arange(0, z_profile_unique_idx_sorted.shape[0], self.step_size_for_control_pts)] # Param (Step size controls number of control points)
            # print("Number of z values is {} for inst-idx {} for cloud of shape {}".format(z_profile.shape[0], count+1, cur_instant_pc_xyz.shape[0]))
            # Quering the KD-Tree

            for i in range(z_profile_percentile_selected_idx.shape[0]):

                cur_control_point = cur_instant_pc_xyz[z_profile_percentile_selected_idx[i], :].reshape(-1,1)
                [k, tree_idx, _] = o_pcd_tree.search_radius_vector_3d(cur_control_point, self.kdtree_search_radius) # Param (Search radius to link points to control points)
                cur_tree_slash_points = cur_instant_pc_xyz[tree_idx]
                cur_tree_slash_filtered = cur_tree_slash_points[abs(cur_tree_slash_points[:, 2]-cur_control_point[2]) <= self.slash_height_thresh, :] # Param (Controls tree ring formation)
                tree_idx_slash = np.asarray(tree_idx)[abs(cur_tree_slash_points[:, 2]-cur_control_point[2]) <= self.slash_height_thresh] # Param (Controls tree ring formation)

                cur_control_point_diameter = self.calc_max_norm_values(cur_tree_slash_filtered)

                if not self.control_points_dict:
                    self.control_points_dict["points"] = []
                else:
                    self.control_points_dict["points"].append([cur_control_point.reshape(-1,)[0], cur_control_point.reshape(-1,)[1], cur_control_point.reshape(-1,)[2], cur_control_point_diameter])
                # print("For {} instance and control point {}, diameter is {}".format(count+1, i+1, cur_control_point_diameter))
                o_pcd.colors[z_profile_percentile_selected_idx[i]] = [1, 0, 0]
                np.asarray(o_pcd.colors)[tree_idx_slash[1:], :] = [0, 1, 0]
                # o3d.visualization.draw_geometries([o_pcd])
                # o_pcd.paint_uniform_color([0.3, 0.3, 0.3])

            # np.asarray(o_pcd.colors)[z_profile_percentile_selected_idx, :] = [1, 0, 0]
            if self.show_slashes:
                o3d.visualization.draw_geometries([o_pcd])

        # with open("control_points.json", "w") as ofile:
        #     json.dump(self.control_points_dict, ofile)

        if (cur_timestamp.to_sec() - self.last_time_stamp.to_sec()) > self.control_pts_cluster_time: # Param (When to cluster accumulated control points)
            
            self.last_time_stamp = cur_timestamp
            self.cluster_control_points(visualize=self.viz_clustered_control_points)
    
    def cluster_centroid_check(self, centroid, cluster_centroids, threshold=1.0):
        """
        L2 distance check of cluster centroids
        centroid is np.array of (3,)
        cluster_centroids is a list of centroids
        Check the L2 distance is at least threshold meters apart
        """
        # print("Checking centroid {}".format(centroid))
        if len(cluster_centroids) == 0:
            return True

        for cluster_centroid in cluster_centroids:
            distance = np.linalg.norm(centroid - cluster_centroid)
            if distance < threshold:
                print("Failed distance check between {} and {}".format(centroid, cluster_centroid))
                return False
        
        return True


    def cluster_control_points(self, visualize=True):

        self.cluster_points_dict = {}
        points_with_dia = np.array(self.control_points_dict["points"]) # 4th column is the diameter
        points_xyz = points_with_dia[:, :3]
        print("\n----------Control Point Clustering-------------\n")
        print("Num of full control points {}".format(points_xyz.shape[0]))

        if visualize:

            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.5, origin=[0, 0, 0])
            o_pcd = o3d.geometry.PointCloud()
            o_pcd.points = o3d.utility.Vector3dVector(points_xyz)
            o_pcd.paint_uniform_color([0.0, 0.0, 0.0])
            # o3d.visualization.draw_geometries([o_pcd, mesh_frame])

        points_xyz_cluster_labels = self.cluster_tree(points_xyz, min_samples=self.control_pt_min_sample_dbscan, eps=self.control_pt_eps_dbscan, use_3d=True) # Param (DBScan for control points)
        num_of_clusters = np.unique(points_xyz_cluster_labels)

        cluster_centroids = []
        filter_indices = np.array([])

        # if visualize:

        #     if self.perma_clus_color.shape[0] == 0 or num_of_clusters.shape[0]<self.perma_clus_color.shape[0]:
        #         clus_color = self.gen_rand_colors(num_of_clusters.shape[0])
        #         self.perma_clus_color = clus_color
        #     else:
        #         print(num_of_clusters.shape[0])
        #         print(self.perma_clus_color.shape[0])
        #         clus_color = self.gen_rand_colors(num_of_clusters.shape[0] - self.perma_clus_color.shape[0])
        #         self.perma_clus_color = np.vstack((self.perma_clus_color, clus_color))

        for clus_num, clus_label in enumerate(num_of_clusters):

            if clus_label== -1:
                filter_indices = np.append(filter_indices, np.where(points_xyz_cluster_labels == clus_label)[0])
                continue
            
            cur_cluster_idx = points_xyz_cluster_labels==clus_label
            centroid = np.mean(points_xyz[cur_cluster_idx, :],axis=0)

            if cur_cluster_idx.sum() < self.control_pt_min_cluster_pts: # Param (min points to be considered as valid control point cluster)
                print("Points were only {} in clus {}. Skipped!!!".format(cur_cluster_idx.sum(), clus_label))
                filter_indices = np.append(filter_indices, np.where(points_xyz_cluster_labels == clus_label)[0])
                continue
            elif self.cluster_centroid_check(centroid, cluster_centroids) == False:
                print("skip {} pts in clus {}".format(cur_cluster_idx.sum(), clus_label ))
                filter_indices = np.append(filter_indices, np.where(points_xyz_cluster_labels == clus_label)[0])
                continue
            else:
                cluster_centroids.append(centroid)
                clus_points_to_save= np.hstack((points_with_dia[cur_cluster_idx, :], np.array([clus_label]*cur_cluster_idx.sum()).reshape(-1,1)))
                if not self.cluster_points_dict:
                    self.cluster_points_dict["points"] = clus_points_to_save.tolist()
                    with open("tree_cluster_points.json", "w") as ofile:
                        json.dump(self.cluster_points_dict, ofile)
                else:
                    self.cluster_points_dict["points"] = np.vstack((self.cluster_points_dict["points"], clus_points_to_save)).tolist()
                    with open("tree_cluster_points.json", "w") as ofile:
                        json.dump(self.cluster_points_dict, ofile)

                if visualize:
                    color_name = self.get_color_name(self.perma_clus_color[clus_num, :])
                    print("Num of valid control points {} in clus {} with color {}".format(cur_cluster_idx.sum(), clus_label, color_name))
                else:
                    print("Num of valid control points {} in clus {}".format(cur_cluster_idx.sum(), clus_label))
                # self.gen_diameter_profile()

                if visualize:

                    # np.asarray(o_pcd.colors)[cur_cluster_idx, :] = list(clus_color[clus_num, :])
                    np.asarray(o_pcd.colors)[cur_cluster_idx, :] = list(self.perma_clus_color[clus_num, :])

        print("\n------------------------------------------------\n")

        if visualize and self.cluster_points_dict:
            
            self.display_picked_points(o_pcd, points_xyz_cluster_labels)
            self.display_valid_trees(o_pcd, points_xyz, filter_indices)
            self.gen_diameter_profile()

            while True:
                user_input = input("Enter command (p: pick points, d: generate diameter profile, v: display valid trees, s: save tree pcds, m: show map of indices, c: continue): ")

                if user_input == 'p':
                    self.display_picked_points(o_pcd, points_xyz_cluster_labels)
                elif user_input == 'd':
                    self.gen_diameter_profile()
                elif user_input == 'v':
                    self.display_valid_trees(o_pcd, points_xyz, filter_indices)
                elif user_input == 's':
                    self.save_tree_pcds(o_pcd, points_xyz, filter_indices, points_xyz_cluster_labels)
                elif user_input == 'm':
                    self.save_indices_2d_map(o_pcd, points_xyz, filter_indices, points_xyz_cluster_labels)
                elif user_input == 'c':
                    self.save_tree_pcds(o_pcd, points_xyz, filter_indices, points_xyz_cluster_labels)
                    self.save_indices_2d_map(o_pcd, points_xyz, filter_indices, points_xyz_cluster_labels)
                    break
                else:
                    print("Invalid command. Please enter a valid command.")

            # Display valid trees before quitting
            # import pdb; pdb.set_trace()
    def save_tree_pcds(self, o_pcd, points_xyz, filter_indices, points_xyz_cluster_labels):

        # Check if the directory exists, overwrite if user permits
        pcds_path = os.path.join(self.data_path, "pcds")
        if not os.path.exists(pcds_path):
            os.makedirs(pcds_path)
            print(f"Directory '{pcds_path}' created.")
        else:
            response = input(f"Directory '{pcds_path}' already exists. Do you want to overwrite its contents? (y/n): ")
            
            if response.lower() == "y":
                # Remove all files in the directory
                for filename in os.listdir(pcds_path):
                    file_path = os.path.join(pcds_path, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                print(f"Contents of directory '{pcds_path}' cleared.")
            else:
                print("Do not delete. Exit save tree pcds function.")
                return


        labeled_colors = np.asarray(o_pcd.colors)
        filtered_labeled_colors = labeled_colors[~np.in1d(np.arange(labeled_colors.shape[0]), filter_indices)]
        filtered_points_xyz = points_xyz[~np.in1d(np.arange(points_xyz.shape[0]), filter_indices)]

        # Loop through the unique colors for each cluster, save PCD accordingly
        for color in np.unique(filtered_labeled_colors, axis=0):
            cluster_mask = np.all(filtered_labeled_colors == color, axis=1)
            cluster_points = filtered_points_xyz[cluster_mask]
            idx = np.where(np.all(points_xyz==cluster_points[0],axis=1))[0][0]
            label = points_xyz_cluster_labels[idx]

            tree_pcd = o3d.geometry.PointCloud()
            tree_pcd.points = o3d.utility.Vector3dVector(filtered_points_xyz[cluster_mask])
            tree_pcd_file = os.path.join(pcds_path, f"tree{label}.pcd")
            o3d.io.write_point_cloud(tree_pcd_file, tree_pcd)
            print(f"PCD file '{tree_pcd_file} with {filtered_points_xyz[cluster_mask].shape[0]} pts' saved.")

    def display_valid_trees(self, o_pcd, points_xyz, filter_indices):
            # Only visualize the valid trees
            labeled_colors = np.asarray(o_pcd.colors)
            filtered_labeled_colors = labeled_colors[~np.in1d(np.arange(labeled_colors.shape[0]), filter_indices)]
            filtered_points_xyz = points_xyz[~np.in1d(np.arange(points_xyz.shape[0]), filter_indices)]
            
            o_pcd = o3d.geometry.PointCloud()
            o_pcd.points = o3d.utility.Vector3dVector(filtered_points_xyz)
            o_pcd.paint_uniform_color([0.0, 0.0, 0.0])
            np.asarray(o_pcd.colors)[:]  = filtered_labeled_colors
            o3d.visualization.draw_geometries([o_pcd])

    def save_indices_2d_map(self, o_pcd, points_xyz, filter_indices, points_xyz_cluster_labels):
            # Only visualize the valid trees
            labeled_colors = np.asarray(o_pcd.colors)
            filtered_labeled_colors = labeled_colors[~np.in1d(np.arange(labeled_colors.shape[0]), filter_indices)]
            filtered_points_xyz = points_xyz[~np.in1d(np.arange(points_xyz.shape[0]), filter_indices)]

            o_pcd = o3d.geometry.PointCloud()
            o_pcd.points = o3d.utility.Vector3dVector(filtered_points_xyz)
            o_pcd.paint_uniform_color([0.0, 0.0, 0.0])
            np.asarray(o_pcd.colors)[:]  = filtered_labeled_colors

            # Convert the 3D points to 2D coordinates for visualization
            valid_points_2d = np.array(o_pcd.points)[:, :2]
            
            # Create a 2D image with Matplotlib
            fig, ax = plt.subplots()
            ax.scatter(valid_points_2d[:, 0], valid_points_2d[:, 1], c=filtered_labeled_colors, s=10, cmap='viridis')

            # Add cluster labels as text at cluster centroids
            for color in np.unique(filtered_labeled_colors, axis=0):
                cluster_mask = np.all(filtered_labeled_colors == color, axis=1)
                cluster_points = filtered_points_xyz[cluster_mask]
                idx = np.where(np.all(points_xyz==cluster_points[0],axis=1))[0][0]
                label = points_xyz_cluster_labels[idx]
                centroid = np.mean(cluster_points[:, :2], axis=0)
                ax.text(centroid[0], centroid[1], str(label), fontsize=12, color='black')

            if self.traj_file is not None:
                with open(self.traj_file, "r") as file:
                    lines = file.readlines()

                xy_coordinates = []

                for line in lines[1:]:  # Skip the header line
                    parts = line.split()
                    x = float(parts[1])
                    y = float(parts[2])
                    xy_coordinates.append((x, y))

                # Convert trajectory coordinates to image indices
                trajectory_indices = []
                for x, y in xy_coordinates:
                    # row = int((y - y_min) / resolution)
                    # col = int((x - x_min) / resolution)
                    trajectory_indices.append((y, x))

                # Plot the trajectory on the top-down view image
                for i in range(1, len(trajectory_indices)):
                    plt.plot([trajectory_indices[i-1][1], trajectory_indices[i][1]], [trajectory_indices[i-1][0], trajectory_indices[i][0]], color='red', linewidth=0.5)

            # Check if the directory exists, overwrite if user permits
            # Save the 2D image to the specified file path
            png_path = os.path.join(self.data_path, "png")
            if not os.path.exists(png_path):
                os.makedirs(png_path)
                print(f"Directory '{png_path}' created.")
            else:
                response = input(f"Directory '{png_path}' already exists. Do you want to overwrite its contents? (y/n): ")
                
                if response.lower() == "y":
                    plt.savefig(os.path.join(png_path, "trees.png"), dpi=300, bbox_inches='tight')
                    print(f"Saved trees.png")
                else:
                    print("Do not delete.")

            # Display the 2D image
            plt.show()

    def display_picked_points(self, o_pcd, clus_labels):
        # Points are (x,y,z,diameter,cluster)
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(o_pcd)
        vis.run()
        vis.destroy_window()
        picked_pts_idx = vis.get_picked_points()

        clus_label_vals = clus_labels[picked_pts_idx]

        print("\n******************************\n")
        print("Selected Cluster Labels Are {}".format(np.unique(clus_label_vals)))
        print("\n******************************\n")

    def gen_diameter_profile(self):

        try:
            control_points = json.load(open("tree_cluster_points.json"))
        except Exception:
            print("Json file not present!! See if it is saved")

        control_points = np.array(control_points["points"])

        num_clusters = np.unique(control_points[:, 4])

        self.dbh_data = np.zeros((len(num_clusters), 2))
        indices_dict = {value: value for value in num_clusters}
        # Save data in converted indices
        if self.converted_indices_filename is not None:
            indices_dict = {}
            with open(self.converted_indices_filename, 'r') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    value = int(row[0].strip())
                    key = int(row[1].strip())
                    indices_dict[key] = value


        print("\n---------------Tree Diameter Profile-----------\n")
        dbh_dict = {}
        dbh_not_enough_pts = 0
        dbh_over_max = 0
        for clus_label in num_clusters:
            cur_cluster_diameter_stats = []
            cur_cluster_idx = control_points[:, 4]==clus_label
            print("\n*********Processing cluster {} which has {} points***********\n".format(clus_label, cur_cluster_idx.sum()))

            cur_cluster_control_pts = control_points[cur_cluster_idx, :]

            # Fixing height
            min_height_cor_pt = np.min(cur_cluster_control_pts[:, 2])
            if min_height_cor_pt < 0:
                cur_cluster_control_pts[:, 2] = cur_cluster_control_pts[:, 2] + abs(min_height_cor_pt)
            else:
                cur_cluster_control_pts[:, 2] = cur_cluster_control_pts[:, 2] - min_height_cor_pt
            
            print("Tree height is {:.3f}m\n".format(np.max(cur_cluster_control_pts[:, 2]) - np.min(cur_cluster_control_pts[:, 2])))

            max_tree_point = np.percentile(cur_cluster_control_pts[:, 2], 99) # TODO Also implement for min and max
            min_tree_point = np.percentile(cur_cluster_control_pts[:, 2], 1)
            num_profile_segments = int((abs(max_tree_point - min_tree_point))/self.per_tree_profile_height) # Param (Distance per tree diameter profile)
            # z_profile_heights = np.linspace(min_tree_point, max_tree_point, num_profile_segments)
            z_profile_heights = np.arange(0, max_tree_point, self.step_size_for_tree_diameters)
            dbh_dia = self.get_DBH(cur_cluster_control_pts)

            if dbh_dia is not None:
                if dbh_dia[0] > self.max_dbh:
                    dbh_over_max += 1
                    dbh_dia = None
            else:
                dbh_not_enough_pts += 1

            for cur_z_height in z_profile_heights:

                cur_control_slash_idx = abs(cur_cluster_control_pts[:, 2] - cur_z_height) <= self.slash_height_thresh #self.slash_height_thresh # Param (height thresh for ring generation)

                if cur_control_slash_idx.sum() < self.min_slash_points_for_median_diameter: # Param (min number of slash points for diameter to be medianed at that control point)

                    print("Not enough slash point ({}) at that height {:.3f}m so skipping this!!".format(cur_control_slash_idx.sum(), cur_z_height)) 
                else:

                    cur_control_slash_pts = cur_cluster_control_pts[cur_control_slash_idx, :]
                    # Editing how the final diameter is estimated based on noise
                    # if cur_z_height <= 5.0: 
                    #     cur_height_diameter_median = np.percentile(cur_control_slash_pts[:, 3], 75)
                    # else:
                    #     cur_height_diameter_median = np.percentile(cur_control_slash_pts[:, 3], 75)
                    
                    # if abs(np.median(cur_control_slash_pts[:, 3]) - np.max(cur_control_slash_pts[:, 3])) > 0.1: 
                    #     cur_height_diameter_median = np.percentile(cur_control_slash_pts[:, 3], 65)
                    # else:
                    #     cur_height_diameter_median = np.percentile(cur_control_slash_pts[:, 3], 75)
                    
                    ######################################################################
                    # New method for calculating diameter
                    #######################################################################
                    cur_height_diameter_median = np.percentile(cur_control_slash_pts[:, 3], self.dbh_percentile) # hard code to 85th percentile
                    if dbh_dia is not None:
                        if cur_height_diameter_median > dbh_dia[0]:

                            new_control_points = cur_control_slash_pts[cur_control_slash_pts[:, 3]<dbh_dia[0], 3]
                            new_control_points = np.append(new_control_points, [dbh_dia[0]])
                            cur_height_diameter_median = np.percentile(new_control_points, self.dbh_percentile)

                    cur_height_diameter_std = np.std(cur_control_slash_pts[:, 3])
                    # print("At Height {:.3f}m, Median Diameter is {:.3f}m and the STD is {:.3f}m and full array is {}".format(cur_z_height, cur_height_diameter_median, cur_height_diameter_std, np.sort(cur_control_slash_pts[:, 3])))
                    print("At Height {:.3f}m, Median Diameter is {:.3f}m and the STD is {:.3f}m".format(cur_z_height, cur_height_diameter_median, cur_height_diameter_std))
                    cur_cluster_diameter_stats.append((cur_z_height, cur_height_diameter_median, cur_height_diameter_std))

            cur_cluster_diameter_stats = np.array(cur_cluster_diameter_stats)
            if dbh_dia is not None:
                print("\n Median DBH is {:.3f}m and STD DBH is {:.3f}\n and full array is {}".format(dbh_dia[0], dbh_dia[1], np.sort(dbh_dia[2])))
                dbh_dict[clus_label] = dbh_dia[0]
            else:
                print("No DBH Obtained!")

            print("\nMean Full Tree Diameter is {:.3f}m and STD Full Tree Diameter is {:.3f}m".format(np.mean(cur_cluster_diameter_stats[:, 1]), np.std(cur_cluster_diameter_stats[:, 1])))

            if self.plot_tree_profile:
                self.gen_tree_profile_plot(cur_cluster_diameter_stats, clus_label)
            
            if self.save_stats_as_df:
                self.save_as_df(cur_cluster_diameter_stats, clus_label)
            
            # un-comment later
            # if clus_label in indices_dict:
            #     self.dbh_data[indices_dict[clus_label]] = np.array([dbh_dia[0], dbh_dia[1]])

            print("\n**********************************************\n")
        
        print("\n------------------------------------------------------\n")

        # Print the DBH dictionary with formatted keys and values
        avg_dbh = sum(dbh_dict.values()) / len(dbh_dict)
        std_dbh = np.std(list(dbh_dict.values()))
        print("\nDBH dict:")
        for key, value in dbh_dict.items():
            formatted_key = int(key)
            formatted_value = "{:.3f}".format(value)
            print("{}: {}".format(formatted_key, formatted_value))
        print("{}/{} trees give valid DBH with mean {:.3f} and std {:.3f}".format(len(dbh_dict), len(num_clusters), avg_dbh, std_dbh))
        print("{} not enough slash pts, {} over DBH max threshold {}".format(dbh_not_enough_pts, dbh_over_max, self.max_dbh))

    def save_as_df(self, stats_arr, clus_num):

        # Stats Arr is (tree_profile_height, median_tree_diameter, std_tree_diameter)
        df = pd.DataFrame(stats_arr, columns=["Tree Profile Height", "Tree Median Diameter", "Tree STD Diameter"])
        df.to_pickle(self.data_path + "tree_"+str(int(clus_num))+"_dataframe.pkl")
        df.to_csv(self.data_path + "tree_"+str(int(clus_num))+"_excel.csv")

        np.savetxt('data/dbh_output.csv', self.dbh_data, delimiter=',')

    def get_DBH(self, control_pts):

        # DHB as defined in the web is at 1.3716m
        # Merced data taken at 12" (0.3048 m)
        cur_control_slash_idx = abs(control_pts[:, 2] - 1.3716) <= self.slash_height_thresh #self.slash_height_thresh

        if cur_control_slash_idx.sum() < self.min_slash_points_for_median_diameter:
            return None
        else:
            control_slash_pts = control_pts[cur_control_slash_idx, :]
            return (np.percentile(control_slash_pts[:, 3], self.dbh_percentile), np.std(control_slash_pts[:, 3]), control_slash_pts[:, 3])

    def gen_tree_profile_plot(self, stats_arr, clus_num):

        # Stats Arr is (tree_profile_height, median_tree_diameter, std_tree_diameter)
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(stats_arr[:, 0], stats_arr[:, 1], label = "Median Diameter")
        plt.plot(stats_arr[:, 0], stats_arr[:, 2], label = "STD of Diameter")
        plt.title("Tree Diameter Profile")
        plt.xlabel("Tree Height")
        plt.ylabel("Tree Diameter")
        plt.legend()
        plt.savefig(self.data_path + "tree_"+str(int(clus_num))+"_diameter_profile.png")

    def gen_rand_colors(self, num_clusters):

        return np.random.uniform(0.0, 1.0, (num_clusters, 3))
    
    # helper function from https://stackoverflow.com/questions/9694165/convert-rgb-color-to-english-color-name-like-green-with-python
    def closest_color(self, requested_color):

        min_color = {}
        for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - requested_color[0]) ** 2
            gd = (g_c - requested_color[1]) ** 2
            bd = (b_c - requested_color[2]) ** 2
            min_color[(rd + gd + bd)] = name

        return min_color[min(min_color.keys())]
    
    def get_color_name(self, color):
        
        color = color.squeeze()*255.0
        color_tuple = tuple(color.astype("int"))

        try:
            actual_name = webcolors.rgb_to_name(color_tuple)
            return actual_name
        except ValueError:
            closest_name = self.closest_color(color_tuple)
            return closest_name


    def publish_clustered_cloud(self, clus_tree_cloud, cur_timestamp, cur_frame_id, publisher):

        pc_msg = PointCloud2()

        header = Header()
        header.stamp = cur_timestamp
        header.frame_id = cur_frame_id
        pc_msg.header = header

        pc_msg.width = clus_tree_cloud.shape[0]
        pc_msg.height = 1
        pc_msg.point_step = 16
        pc_msg.row_step = pc_msg.width * pc_msg.point_step
        pc_msg.fields = self.pc_fields_
        full_data = clus_tree_cloud.astype(np.float32)
        pc_msg.data = full_data.tobytes()
        publisher.publish(pc_msg)

    def cluster_tree(self, pc_xyz, min_samples, eps, n_jobs=5, use_3d=True):

        if use_3d:

            pc_xyz = pc_xyz[:, :3]

        else:

            pc_xyz = pc_xyz[:, :2]

        optics_obj = DBSCAN(min_samples=min_samples,
                            eps=eps, n_jobs=n_jobs).fit(pc_xyz)

        pc_labels = optics_obj.labels_

        return pc_labels

    def threshold_by_range(self, valid_range_threshold, pc_xyzi):

        points_pano_xyz = pc_xyzi[:, :2]
        range_values = np.linalg.norm(points_pano_xyz, axis=1)
        # filter out points at 0
        valid_indices = (range_values >= 0.1)
        # filter out points larger than valid_range_threshold
        valid_indices = np.logical_and(
            (range_values < valid_range_threshold), valid_indices)

        return valid_indices
    
    def calc_max_norm_values(self, vals):

        max_norm = 0

        for i in range(vals.shape[0]):

            diff_vec = vals - vals[i]
            max_dia = np.max(np.linalg.norm(diff_vec, axis=1))

            if max_dia > max_norm:

                max_norm = max_dia 

        return max_norm

def make_fields():
    # manually add fiels by Ian
    fields = []
    field = PointField()
    field.name = 'x'
    field.count = 1
    field.offset = 0
    field.datatype = PointField.FLOAT32
    fields.append(field)

    field = PointField()
    field.name = 'y'
    field.count = 1
    field.offset = 4
    field.datatype = PointField.FLOAT32
    fields.append(field)

    field = PointField()
    field.name = 'z'
    field.count = 1
    field.offset = 8
    field.datatype = PointField.FLOAT32
    fields.append(field)

    field = PointField()
    field.name = 'intensity'
    field.count = 1
    field.offset = 12
    field.datatype = PointField.FLOAT32
    fields.append(field)
    return fields

if __name__ == "__main__":

    rospy.init_node("tree_cloud_node")
    tree_node = ProcessTreeNode()
    while not rospy.is_shutdown():

        print("Tree Node Started")
        rospy.spin()

    if tree_node.compute_tree_profile:
        tree_node.gen_diameter_profile()
    print("Node Killed")
