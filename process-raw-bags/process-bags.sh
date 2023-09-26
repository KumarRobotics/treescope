#!/bin/bash
dataset_json='json/VAT-1022.json'

source /home/derek/Workspaces/backpack_ws/devel/setup.bash


process_bag() {
  echo "Dataset: $1"
  echo "Raw Bag: $2"
  echo "Bag: $3"
  echo "Lidar: $4"
  echo "OS Metadata: $5"
  echo "Model Path: $6"
  echo "--------------------"

  # Run rosbag info command and extract desired information
  info_output=$(rosbag info --yaml /derek_data/TreeScope/$1/input/$2)
  duration=$(echo "$info_output" | awk '/duration:/{print $2}')
  echo $duration

  # Switch to correct branch and build
  roscd scan2shape_launch/script/
  if [ $4 == "OS0-128" ]; then
    git checkout factor_graph_dbh_orchard_os0-128
  elif [ $4 == "OS1-64" ]; then
    git checkout factor_graph_dbh_orchard
  fi
  catkin build

  # Run process_cloud_node.py in the background
  python /home/derek/Workspaces/sloam_ws/src/generic-sloam/scan2shape/script/process_cloud_node.py &
  cloud_node_pid=$!
  sleep 1
f
  # Change directory and run infer_node.py in the background
  cd /home/derek/Workspaces/sloam_ws/src/generic-sloam/scan2shape/script/ && python infer_node.py -d "$6" &
  infer_node_pid=$!
  sleep 1

  if [ $4 == "OS0-128" ]; then

    # Run fasterlio
    roslaunch ouster_ros replay.launch metadata:=/derek_data/TreeScope/$1/metadata/$5 & 
    replay_pid=$!

    roslaunch faster_lio mapping_ouster128.launch & 
    fasterlio_pid=$!
    sleep 5

    # Record bag
    rosbag record -O "/derek_data/TreeScope/$1/processed/$3" /Odometry /tf /tf_static /os_node/segmented_point_cloud_no_destagger /tree_cloud /tree_cloud_world /ground_cloud /cloud_registered_body /ublox/fix /ublox/fix_velocity __name:=record_bag &
    sleep 1

    # Finally play the bag
    rosbag play "/derek_data/TreeScope/$1/input/$2" /ouster/imu_packets:=/os_node/imu_packets /ouster/lidar_packets:=/os_node/lidar_packets  --start 0.1
    playbag_pid=$!

  
  elif [ $4 == "OS1-64" ]; then
    # Run fasterlio
    roslaunch ouster_ros replay.launch & 
    replay_pid=$!

    roslaunch faster_lio mapping_ouster64.launch & 
    fasterlio_pid=$!
    sleep 5

    # Record bag
    rosbag record -O "/derek_data/TreeScope/$1/processed/$3" /Odometry /tf /tf_static /os_node/segmented_point_cloud_no_destagger /tree_cloud /tree_cloud_world /ground_cloud /cloud_registered_body /ublox/fix /ublox/fix_velocity __name:=record_bag &
    sleep 1

    # Finally play the bag
    rosbag play "/derek_data/TreeScope/$1/input/$2" --start 0.1 --topics /os_node/lidar_packets /os_node/imu_packets /os_node/metadata /ublox/fix /ublox/fix_velocity
    playbag_pid=$!
  
  fi

  # Sleep duraiton + 5
  # sleep $duration
  echo "Killing processes $cloud_node_pid $infer_node_pid $fasterlio_pid $playbag_pid $replay_pid"
  kill -INT $cloud_node_pid $infer_node_pid $fasterlio_pid $playbag_pid $replay_pid
  rosnode kill /record_bag
  sleep 10

  echo "Done with $3"

}


# Read the JSON file and parse its content
datasets=() # Initialize the 2D array datasets
while IFS= read -r line; do
  if [[ "$line" =~ "}," ]]; then
    # Append the current dataset to the datasets array
    datasets+=("${dataset_array[@]}")

  elif [[ "$line" =~ "dataset\":" ]]; then
    dataset_array[0]=$(echo "$line" | awk '{print $2}' | tr -d ',"')
  elif [[ "$line" =~ "raw_bag\":" ]]; then
    dataset_array[1]=$(echo "$line" | awk '{print $2}' | tr -d ',"')
  elif [[ "$line" =~ "bag\":" ]]; then
    dataset_array[2]=$(echo "$line" | awk '{print $2}' | tr -d ',"')
  elif [[ "$line" =~ "lidar\":" ]]; then
    dataset_array[3]=$(echo "$line" | awk '{print $2}' | tr -d ',"')
  elif [[ "$line" =~ "metadata\":" ]]; then
    dataset_array[4]=$(echo "$line" | awk '{print $2}' | tr -d ',"')
  elif [[ "$line" =~ "model\":" ]]; then
    dataset_array[5]=$(echo "$line" | awk '{print $2}' | tr -d ',"')
  fi
done < $dataset_json

# Iterate through each dataset and print its contents
for ((i = 0; i < ${#datasets[@]}; i+=6)); do
  process_bag "${datasets[i]}" "${datasets[i+1]}" "${datasets[i+2]}" "${datasets[i+3]}" "${datasets[i+4]}" "${datasets[i+5]}"
done
