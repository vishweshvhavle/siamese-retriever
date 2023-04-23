import os
import random
import csv

# Path to your directories
train_dir = '../data/train/images'
train_csv = '../data/train/train_labels.csv'
test_dir = '../data/test/images'
test_csv = '../data/test/test_labels.csv'
val_dir = '../data/val/images'
val_csv = '../data/val/val_labels.csv'

# Define a function to shuffle the files in a directory
def shuffle_files(directory_path, csv_path):
    # Get a list of JPG files in the directory
    jpg_files = [f for f in os.listdir(directory_path) if f.endswith('.jpg')]

    # Randomly shuffle the list of JPG files
    random.shuffle(jpg_files)

    # Create a dictionary to store the mapping between old and new file names
    file_mapping = {}
    for i, f in enumerate(jpg_files):
        old_path = os.path.join(directory_path, f)
        new_path = os.path.join(directory_path, f"{i+1}.jpg")
        while os.path.exists(new_path):
            # If a file with the new name already exists, add a suffix to the file name
            suffix = random.randint(1, 9999)
            new_path = os.path.join(directory_path, f"{i+1}_{suffix}.jpg")
        file_mapping[old_path] = new_path
        os.rename(old_path, new_path)

    # Update the CSV file with the new file names
    if os.path.isfile(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for row in rows:
                old_path = os.path.join(directory_path, row[0])
                new_path = file_mapping[old_path]
                row[0] = os.path.basename(new_path)  # Update the file name
                writer.writerow(row)

# Shuffle the files in each directory and update the CSV files
shuffle_files(train_dir, train_csv)
shuffle_files(test_dir, test_csv)
shuffle_files(val_dir, val_csv)
