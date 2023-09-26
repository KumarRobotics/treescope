import time
import argparse
from train_test_split import train_test_split_save
from convert_images_to_labels_pcds import convert_images_to_labels_pcds

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
