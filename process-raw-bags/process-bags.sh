#!/bin/bash
dataset_json='/derek_data/dataset/lidar-dataset-UCM-0323.json'

source /home/derek/Workspaces/catkin_ws/devel/setup.bash


process_bag() {
  echo "Dataset: $1"
  echo "Raw Bag: $2"
  echo "Bag: $3"
  echo "Lidar: $4"
  echo "Model Path: $5"
  echo "--------------------"

  # Run rosbag info command and extract desired information
  info_output=$(rosbag info --yaml /derek_data/dataset/bags/$1/$2)
  duration=$(echo "$info_output" | awk '/duration:/{print $2}')
  echo $duration

  # Switch to correct branch and build
  roscd scan2shape_launch/script/
  if [ $4 == "OS0-128" ]; then
    git checkout devel/orchard_os0-128
  elif [ $4 == "OS1-64" ]; then
    git checkout devel/orchard_os1-64
  fi
  catkin build

  # Run process_cloud_node.py in the background
  python /home/derek/Workspaces/sloam_ws/src/generic-sloam/scan2shape/script/process_cloud_node.py &
  cloud_node_pid=$!
  sleep 1
f
  # Change directory and run infer_node.py in the background
  cd /home/derek/Workspaces/sloam_ws/src/generic-sloam/scan2shape/script/ && python infer_node.py -d "$5" &
  infer_node_pid=$!
  sleep 1

  if [ $4 == "OS0-128" ]; then
  echo "128"
    # Run fasterlio
    roslaunch scan2shape_launch run_flio_with_driver.launch replay:=true metadata:=/derek_data/dataset/UCM-OS0-128-SensorTower.json use_ouster128:=true &
    fasterlio_pid=$!
    sleep 5

    # Record bag
    rosbag record -O "/derek_data/dataset/bags/$1/$3" /Odometry /tf /tf_static /os_node/segmented_point_cloud_no_destagger /tree_cloud /ground_cloud __name:=record_bag &
    sleep 1

    # Finally play the bag
    rosbag play "/derek_data/dataset/bags/$1/$2" /ouster/imu_packets:=/os_node/imu_packets /ouster/lidar_packets:=/os_node/lidar_packets  --start 0.1
    playbag_pid=$!

  
  elif [ $4 == "OS1-64" ]; then
    # Run fasterlio
    roslaunch scan2shape_launch run_flio_with_driver.launch replay:=true &
    fasterlio_pid=$!
    sleep 5

    # Capture the bag and run the required commands
    rosbag record -O "/derek_data/dataset/bags/$1/$3" /Odometry /tf /tf_static /os_node/segmented_point_cloud_no_destagger /tree_cloud /ground_cloud __name:=record_bag &
    sleep 1

    # # Finally play the bag
    rosbag play "/derek_data/dataset/bags/$1/$2" --start 0.1
    playbag_pid=$!
  
  fi

  # Sleep duraiton + 5
  # sleep $duration
  echo "Killing processes $cloud_node_pid $infer_node_pid $fasterlio_pid $playbag_pid"
  kill -INT $cloud_node_pid $infer_node_pid $fasterlio_pid $playbag_pid
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
  elif [[ "$line" =~ "model\":" ]]; then
    dataset_array[4]=$(echo "$line" | awk '{print $2}' | tr -d ',"')
  fi
done < $dataset_json

# Iterate through each dataset and print its contents
for ((i = 0; i < ${#datasets[@]}; i+=5)); do
  process_bag "${datasets[i]}" "${datasets[i+1]}" "${datasets[i+2]}" "${datasets[i+3]}" "${datasets[i+4]}"
done
