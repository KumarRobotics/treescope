import time
import argparse
import shutil
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import re
import cv2
from pypcd import pypcd
from termcolor import colored
import traceback

def train_test_split_save(for_jackle, for_indoor, args):
    if for_jackle:
        # Maybe dataset_name is redundant and can be removed
        dataset_name = args["data_dir"].split("/")[-2]
        label_prefix = "label_pano" #"label_sweep"
        print("Confirm that this is correct: prefix for the LIDAR data is :", label_prefix)
        time.sleep(2)
    else:
        if for_indoor:
            dataset_name = args["data_dir"].split("/")[-2]
            label_prefix = "label"
        else:
            dataset_name = args["data_dir"].split("/")[-2]
            label_prefix = "label"
    cloud_dir = args["data_dir"]+"converted_scans/"
    label_dir = args["data_dir"]+"converted_labels/"
    save_dir = args["data_dir"]
    clouds = glob.glob(cloud_dir + 'point_cloud_*.pcd')
    test_portion = 0.1
    print("percentage of data splited as test set: ", test_portion)
    train, test = train_test_split(clouds, test_size=test_portion, random_state=42)

    output_sequence_dir = save_dir + "sequences/"
    if os.path.exists(output_sequence_dir):
        print("output_sequence_dir already exists, we are going to remove it, which are: \n" + output_sequence_dir)
        input("Press Enter to continue...")
        shutil.rmtree(output_sequence_dir)
    os.makedirs(output_sequence_dir + "00/labels/")
    os.makedirs(output_sequence_dir + "01/labels/")
    os.makedirs(output_sequence_dir + "00/point_clouds/")
    os.makedirs(output_sequence_dir + "01/point_clouds/")
    os.makedirs(output_sequence_dir + "02/point_clouds/")



    for file in train:
        print("loading lables for file: ", file)
        number = re.findall(r'[0-9]+', file)[0]
        shutil.copy(file, save_dir + "sequences/00/point_clouds/point_cloud_" + number + ".pcd")
        label = label_dir + label_prefix + "_" + number + ".npy"
        shutil.copy(label, save_dir + "sequences/00/labels/label_" +number+".npy")
        print("successfully loaded lables for training file: ", file)
    for file in test:
        print("loading lables for file: ", file)
        number = re.findall(r'[0-9]+', file)[0]
        shutil.copy(file, save_dir + "sequences/01/point_clouds/point_cloud_" + number + ".pcd")
        label = label_dir + label_prefix + "_" + number + ".npy"
        shutil.copy(label, save_dir + "sequences/01/labels/label_" +number+".npy")
        print("successfully loaded lables for validation file: ", file)
    for file in test:
        # TODO: test data should probably be created separately from validation set
        print("loading lables for file: ", file)
        number = re.findall(r'[0-9]+', file)[0]
        shutil.copy(file, save_dir + "sequences/02/point_clouds/point_cloud_" + number + ".pcd")
        print("successfully loaded lables for test file: ", file)

def convert_images_to_labels_pcds(for_jackle, for_indoor, args):
    if for_jackle:
        data_dir = args["data_dir"]
    else:
        if for_indoor:
            data_dir = args["data_dir"]
        else:
            data_dir = args["data_dir"]

    if for_jackle:
        prefix = "pano" # "sweep"
        fnames = glob.glob(data_dir + "labels/"+prefix+"*.png") 
        print("Confirm that this is correct: prefix for the LIDAR data is :", prefix)
        time.sleep(2)
    else:
        fnames = glob.glob(data_dir + "labels/1*.png") # start with 1 to avoid including the viz_ stuff

    save_dir_point_cloud = data_dir + "converted_scans/"
    save_dir_label = data_dir + "converted_labels/"

    if os.path.exists(save_dir_point_cloud) or os.path.exists(save_dir_label):
        print("save_dir_point_cloud or save_dir_label already exists, we are going to remove them, which are: \n" + save_dir_point_cloud + " \n and: \n" +save_dir_label)
        input("Press Enter to continue...")

    if os.path.exists(save_dir_point_cloud):
        shutil.rmtree(save_dir_point_cloud)
    os.mkdir(save_dir_point_cloud)

    if os.path.exists(save_dir_label):
        shutil.rmtree(save_dir_label)
    os.mkdir(save_dir_label)


    num_classes = 10
    stats = np.zeros((num_classes, len(fnames)))
    file_idx = 0
    pc_size = -1
    d = None

    for fname in fnames:
        fname_no_png = fname.split(".png")[0]
        fname_no_prefix = fname_no_png.split('/')[-1]
        scan_fname = data_dir + "scans/" + fname_no_prefix +".tiff"
        label_fname = data_dir + "labels/" + fname_no_prefix + ".png"
        # print("currently loading labels and range images for scan: ", scan_fname)

        scan = cv2.imread(scan_fname, cv2.IMREAD_UNCHANGED)
        label = cv2.imread(label_fname)

        # all rays that do not have any return will be set as 0, and they are not considered during the back-propagation
        scan = np.nan_to_num(scan, copy=True, nan=0.0, posinf=None, neginf=None)
        # make sure the label for those points belong to "unlabeled" class, which is 0
        index = (scan.sum(axis=2) == 0)
        label_copy = label
        label_copy[index, :] = 0
        if (label.flatten().sum() != label_copy.flatten().sum() ):
            raise Exception("some of the rays without returns have labels, this should not happen!")
        else:
            label = label_copy

        # convert label into expected values
        # Ian's label tool convention:
        # - unlabelled: [0, 0, 0] -- expected value is 0
        # - road: [0, 0, 1]  -- expected value is 1
        # - vegetation: [0, 1, 0]  -- expected value is 2
        # - building: [1, 0, 0]  -- expected value is 3
        # - grass/sidewalk: [0, 0.4, 0]  -- expected value is 4
        # - vehicle: [0, 1, 1]  -- expected value is 5
        # - human: [1, 0, 1]  -- expected value is 6
        # - gravel: [0, 0.5, 0.5]  -- expected value is 7
        # - tree_trunk: [0.5, 0.2, 0.2]  -- expected value is 8
        # - light_pole: [1, 0.5, 0]  -- expected value is 9
        # label_converted = np.zeros((label.shape[0], label.shape[1]))
        label_converted = label[:,:,0]

        convert_labels_into_specific_class = (for_jackle==False)
        if convert_labels_into_specific_class:
            print("converting labels into specific classes (e.g. only keeping car, light pole, trunk, road and background)")
            # input("Press Enter to confirm and continue, otherwise, kill the program and modify convert_images_to_labels_pcds.py ...")
            background_label = 0
            if for_indoor:
                road_label = 0
            else:
                road_label = 1

            # convert labels of non-car into background
            label_converted[label_converted==0] = background_label
            label_converted[label_converted==1] = road_label

            label_converted[label_converted==2] = background_label
            print(f'converting points with vegetation labels into background labels')

            label_converted[label_converted==3] = background_label
            print(f'converting points with building labels into background labels')

            label_converted[label_converted==4] = road_label
            print(f'converting points with grass/sidewalk labels into road labels')

        # stats of number of points to address class imbalance issues
        num_pts = 0
        for i in np.arange(num_classes):
            stats[i, file_idx] = np.sum((label_converted).ravel() == i)
            num_pts += stats[i, file_idx]
        # sanity check: all points should be included in stats:
        if (num_pts!=label_converted.ravel().shape):
            raise Exception("labels are not 0-9, invalid labels exist!!!")
        file_idx+=1


        # convert scan image into pcd data format
        # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
        # pcd = o3d.geometry.PointCloud()
        x = scan[:,:,0].flatten()
        y = scan[:,:,1].flatten()
        z = scan[:,:,2].flatten()
        xyz = np.zeros((x.shape[0],3))
        xyz[:,0] = x
        xyz[:,1] = y
        xyz[:,2] = z
        mean = np.mean(xyz, axis=0)
        # print(mean)
        intensity = scan[:,:,3].flatten()

        # faster approach
        pc = pypcd.make_xyz_point_cloud(xyz)
        pc.pc_data = pc.pc_data.flatten()
        # old_md = pc.get_metadata()
        # new_dt = [(f, pc.pc_data.dtype[f]) for f in pc.pc_data.dtype.fields]
        # new_data = [pc.pc_data[n] for n in pc.pc_data.dtype.names]
        md = {'fields': ['intensity'], 'count': [1], 'size': [4],'type': ['F']}

        #################################### Speed up by avoiding creating recarray multiple times, which is very slow #######################################################################
        if pc_size == -1:
            # initialize
            pc_size = len(pc.pc_data)
            d = np.rec.fromarrays((np.random.random(len(pc.pc_data))))
        elif len(pc.pc_data) != pc_size:
            print("different size point cloud received, recreating numpy.recarray, which is very slow!")
            pc_size = len(pc.pc_data)
            d = np.rec.fromarrays((np.random.random(len(pc.pc_data))))
        # else:
        #     # continue using existing d
        #     print("point cloud size is the same, which is good!")
        ###################################################################################################################################################################################

        try:
            newpc = pypcd.add_fields(pc, md, d)
        except:
            traceback.print_exc()
            raise Exception(colored("READ THIS: for this error, just comment out the two lines (should be line 443 and 444) in pypcd.py file!",'green'))

        # new_md = newpc.get_metadata()
        # setting intensity data
        newpc.pc_data['intensity'] = intensity


        # save point cloud as pcd files in converted_scans folder
        point_cloud_final_fname = save_dir_point_cloud + "point_cloud" + "_" + str(fname_no_prefix) + ".pcd"
        newpc.save_pcd(point_cloud_final_fname, compression='binary_compressed')
        # print("pcds are saved in converted_scans folder!")
        # remove intermediate point cloud
        # os.remove(point_cloud_temp_fname)

        # save labels as an 1-d array in converted_labels folder
        label_converted = label_converted.ravel()

        np.save(save_dir_label + "label" + "_" + str(fname_no_prefix) + ".npy", label_converted)
        # print("labels are saved in converted_labels folder!")
        print("finished processing: ", file_idx, " out of ", len(fnames), " files")

    # print out the stats
    print("std values of number of points for each class are: ")
    print(np.std(stats, axis=1))
    print("mean values of number of points for each class are: ")
    print(np.mean(stats, axis=1))
    print("mean values of number points for each class / total points are: ")
    percent_array = np.mean(stats, axis=1) / np.sum(np.mean(stats, axis=1))
    print(percent_array)
    np.savetxt(data_dir + "class_points_divided_by_total_points.txt", percent_array,fmt='%.7f')

    print("total values of number points for each class: ")
    total_array = np.sum(stats, axis=1)
    print(total_array)



if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", "-D", required=True, help="The main dataset directory where the labeled data is stored with the final '/' included")
    args = vars(ap.parse_args())

    for_jackle = False
    for_indoor = False
    convert_images_to_labels_pcds(for_jackle, for_indoor, args)
    print('convert_images_to_labels_pcds finished, starting training and test set split in 2 seconds...')
    time.sleep(2)
    train_test_split_save(for_jackle, for_indoor, args)