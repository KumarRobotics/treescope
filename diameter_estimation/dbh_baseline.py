import json
import yaml
import numpy as np
import csv

dataset = "VAT-0723M"
predictions = "VAT-0723-DBCRE.yaml"

# Load the JSON file
dataset_file = dataset + "-gt.json"
with open(dataset_file, 'r') as json_file:
    json_data = json.load(json_file)

# Load the YAML file
with open(predictions, 'r') as yaml_file:
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

diff = dbh_gt-dbh_est
mae = np.mean(np.abs(diff))
rmse = np.sqrt(np.mean((diff)**2))
mean_gt = np.mean(dbh_gt)
mean_est = np.mean(dbh_est)
print("Dataset {}, Mean DBH GT {:.4f}, Mean DBH Estimate {:.4f}, Mean Abs Err {:.4f}, RMSE {:.4f}".format(dataset, mean_gt, mean_est, mae, rmse))

# Dump into CSV with columns and headers tree_ids, dbh_gt, dbh_est, mae, rmse
# Create a list of lists to hold the data
data = []

# Add headers to the data
data.append(["Tree ID", "DBH GT", "DBH Est", "Error"])

# Combine your numpy arrays and convert them to a list of lists
combined_data = np.column_stack((tree_ids, dbh_gt, dbh_est, np.round(diff,4))).tolist()

# Add the combined data to the main data list
data.extend(combined_data)

# Define the CSV file name
csv_file = "results_" + dataset + ".csv"

# Write the data to the CSV file
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)