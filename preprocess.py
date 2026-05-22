import cv2  # Standard for forensic grayscale
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import os


def finalize_milestone_1_augmented():
    # Load the inventory of 1,400 images
    if not os.path.exists("dataset_inventory.csv"):
        print("❌ Error: dataset_inventory.csv not found!")
        return

    df = pd.read_csv("dataset_inventory.csv")

    X, y = [], []

    # Mapping scanner names to numbers
    scanner_map = {
        "Canon120": 0, "Canon220": 1, "Canon9000": 2,
        "EpsonV370": 3, "EpsonV39": 4, "EpsonV550": 5, "HP": 6
    }

    print("--- 🔄 Milestone 1: Patch Augmentation (Target: 94% Accuracy) ---")

    for index, row in df.iterrows():
        try:
            # Use OpenCV for precise forensic grayscale math
            img = cv2.imread(row['Path'], cv2.IMREAD_GRAYSCALE)

            if img is not None:
                h, w = img.shape

                # Check if image is large enough for the augmentation strategy
                if h < 650 or w < 650:
                    # Fallback: Just take the center if the image is too small
                    crop = cv2.resize(img, (512, 512))
                    X.append(crop.astype(np.float32))
                    y.append(scanner_map[row['Scanner']])
                else:
                    # --- PATCH AUGMENTATION ---
                    # We extract 4 different 512x512 patches from the corners/sides
                    # This gives the model 4x the data to learn the hardware fingerprint
                    patches = [
                        img[50:562, 50:562],  # Top-Left patch
                        img[50:562, w - 562:w - 50],  # Top-Right patch
                        img[h - 562:h - 50, 50:562],  # Bottom-Left patch
                        img[h - 562:h - 50, w - 562:w - 50]  # Bottom-Right patch
                    ]

                    for p in patches:
                        X.append(p.astype(np.float32))
                        y.append(scanner_map[row['Scanner']])
            else:
                print(f"⚠️ Warning: Could not read image at {row['Path']}")

        except Exception as e:
            print(f"Error processing {row['Path']}: {e}")

    # Convert to NumPy arrays
    X = np.array(X)
    y = np.array(y)

    # --- SHUFFLING ---
    # Shuffling is now even more critical so patches from the same image are mixed
    X, y = shuffle(X, y, random_state=42)

    # --- SAVING ---
    np.save("X_final.npy", X)
    np.save("y_final.npy", y)

    print("-" * 30)
    print(f"✅ Preprocessing Augmented!")
    print(f"Total Forensic Samples: {len(X)}")  # Should be around 5,600
    print(f"Array Shape: {X.shape}")
    print("-" * 30)


if __name__ == "__main__":
    finalize_milestone_1_augmented()