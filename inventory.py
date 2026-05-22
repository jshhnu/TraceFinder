import os  # Standard library to interact with the operating system (folders/files)
import pandas as pd  # The go-to tool for data tables and CSV management
from PIL import Image  # Library used to handle and verify image files


def organize_and_inventory(root_download_path):
    # This list defines our 7 'Target Classes'. Any scanner not in this list gets ignored.
    target_scanners = ["Canon120", "Canon220", "Canon9000", "EpsonV370", "EpsonV39", "EpsonV550", "HP"]

    # An empty list where we will store the details (scanner name, path, etc.) of each image
    inventory_data = []

    print(f"--- 🚀 Starting Deep Scan of: {root_download_path} ---")

    # Step 1: os.walk is a 'recursive' tool. It dives into every subfolder inside 'Dataset'.
    # root = the current folder, dirs = subfolders, files = filenames in that folder.
    for root, dirs, files in os.walk(root_download_path):
        for filename in files:

            # Step 2: We only care about .tif or .tiff files (the forensic standard).
            # We convert to .lower() to handle files named .TIF or .tif equally.
            if filename.lower().endswith(('.tif', '.tiff')):

                # Step 3: We need to figure out which scanner produced this file.
                # We do this by checking if any of our 'target_scanners' keywords are in the folder path (root).
                matched_scanner = None
                for scanner in target_scanners:
                    if scanner.lower() in root.lower():
                        matched_scanner = scanner  # Found a match!
                        break  # Stop looking for other scanner names for this specific file

                # If the file belongs to one of our 7 scanners, we save its info.
                if matched_scanner:
                    # Combine the folder path and filename to get the full location on your PC
                    img_path = os.path.join(root, filename)

                    # Forensic scanner noise changes based on DPI (Resolution).
                    # We look for the text '150' or '300' in the folder path to categorize it.
                    dpi_val = "150" if "150" in root else "300" if "300" in root else "Unknown"

                    # Add a dictionary of this image's info to our master list
                    inventory_data.append({
                        "Scanner": matched_scanner,
                        "DPI": dpi_val,
                        "FileName": filename,
                        "Path": os.path.abspath(img_path)  # Get the full computer path (C:\Users\...)
                    })

    # Step 4: Convert our list of dictionaries into a structured DataFrame (Table).
    df = pd.DataFrame(inventory_data)

    # Check if we actually found any images.
    if not df.empty:
        # DATA BALANCING: This is a professional trick.
        # We group by Scanner and DPI, then take only the first 100 images for each.
        # This ensures our AI doesn't get 'biased' by having too much data from one brand.
        df = df.groupby(['Scanner', 'DPI']).head(100).reset_index(drop=True)

        # Save this table to a CSV file so our next script (preprocess.py) can read it.
        df.to_csv("dataset_inventory.csv", index=False)

        # Summary report for the user
        print("\n--- ✅ Inventory Complete ---")
        print(f"Total Images Cataloged: {len(df)}")
        print("\nBreakdown per Scanner:")
        print(df['Scanner'].value_counts())  # Shows exactly how many images per brand we have
    else:
        # If the list is empty, something is wrong with the 'Dataset' folder location.
        print("❌ No .tif files found! Double-check your extract path.")


if __name__ == "__main__":
    # This is the entry point. It tells the script to look inside the folder named 'Dataset'.
    organize_and_inventory("Dataset")