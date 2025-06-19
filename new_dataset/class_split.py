import os
import shutil
import re
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Organize files into subfolders based on class number in filename.")
    parser.add_argument(
        '--folder_path', type=str, required=True,
        help='Path to the folder containing files to organize.'
    )
    parser.add_argument(
        '--pattern_index', type=int, default=3,
        help='Index of the part after splitting filename by underscore that contains class number (default: 3).'
    )
    parser.add_argument(
        '--dry_run', action='store_true',
        help='If set, do not actually move files, only print what would be done.'
    )
    return parser.parse_args()

def organize_folder(folder_path, pattern_index=3, dry_run=False):
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a directory or does not exist.")
        return
    moved_count = 0
    skipped_count = 0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            parts = filename.split('_')
            if len(parts) > pattern_index:
                class_part = parts[pattern_index]
                match = re.search(r'\d+', class_part)
                if match:
                    class_number = match.group()
                    target_folder = os.path.join(folder_path, class_number)
                    if not dry_run:
                        os.makedirs(target_folder, exist_ok=True)
                        destination = os.path.join(target_folder, filename)
                        try:
                            shutil.move(file_path, destination)
                            moved_count += 1
                            print(f"Moved: {filename} -> {target_folder}")
                        except Exception as e:
                            print(f"Failed to move {filename}: {e}")
                    else:
                        print(f"[Dry run] Would move: {filename} -> {target_folder}")
                        moved_count += 1
                else:
                    skipped_count += 1
                    print(f"Skipping '{filename}': no numeric class found in part '{class_part}'")
            else:
                skipped_count += 1
                print(f"Skipping '{filename}': expected at least {pattern_index+1} parts after splitting by '_'")
    print(f"Done. Files moved: {moved_count}. Files skipped: {skipped_count}.")

if __name__ == '__main__':
    args = get_args()
    organize_folder(args.folder_path, pattern_index=args.pattern_index, dry_run=args.dry_run)
