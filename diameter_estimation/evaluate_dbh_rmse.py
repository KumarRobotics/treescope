import sys
import json
import yaml
import numpy as np
import csv

if len(sys.argv) < 3:
    print("Usage: python3 evaluate_dbh_rmse.py <dataset.json> <predictions.yaml> [output.csv]")
    sys.exit(1)

dataset_json_path = sys.argv[1]
predictions_yaml_path = sys.argv[2]
dataset = dataset_json_path.replace("-gt.json", "")

if len(sys.argv) > 3:
    csv_file = sys.argv[3]
else:
    csv_file = "results_" + dataset + ".csv"

with open(dataset_json_path, 'r') as json_file:
    json_data = json.load(json_file)

with open(predictions_yaml_path, 'r') as yaml_file:
    yaml_data = yaml.safe_load(yaml_file)

tree_ids = []
dbh_gt = []
dbh_est = []

for key in yaml_data.keys():
    for tree in yaml_data[key]:
        tree_key = key+"_"+tree
        if tree_key in json_data:
            dbh_gt.append(json_data[tree_key]['DBH'])
            dbh_est.append(yaml_data[key][tree])
            tree_ids.append(tree_key)

dbh_gt = np.array(dbh_gt)
dbh_est = np.array(dbh_est)
tree_ids = np.array(tree_ids)

diff = dbh_gt - dbh_est
mae = np.mean(np.abs(diff))
rmse = np.sqrt(np.mean((diff)**2))
mean_gt = np.mean(dbh_gt)
mean_est = np.mean(dbh_est)
print(f"Dataset {dataset}, Mean DBH GT {mean_gt:.4f}, Mean DBH Estimate {mean_est:.4f}, Mean Abs Err {mae:.4f}, RMSE {rmse:.4f}")

# Save tree_ids, dbh_gt, dbh_est, mae, rmse
data = []
data.append(["Tree ID", "DBH GT", "DBH Est", "Error"])

combined_data = np.column_stack((tree_ids, dbh_gt, dbh_est, np.round(diff,4))).tolist()
data.extend(combined_data)

# Write the data to the CSV file
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)