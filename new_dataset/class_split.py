import os
import shutil
import re

# Path to your image folder
folder_path = 'Test'  # change this to your actual folder path

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):
        parts = filename.split('_')
        # Ensure the filename has at least 4 parts, e.g., "crop_x_class_n"
        if len(parts) >= 4:
            # Get the element at index 3 (the part after "class")
            class_part = parts[3]
            # Extract only the numeric portion (this also removes any file extension)
            match = re.search(r'\d+', class_part)
            if match:
                class_number = match.group()
                # Create the subfolder named after the numeric class if not existing
                target_folder = os.path.join(folder_path, class_number)
                os.makedirs(target_folder, exist_ok = True)

                # Construct full destination file path and move the file
                destination = os.path.join(target_folder, filename)
                shutil.move(file_path, destination)

print("Files have been organized into subfolders!")
