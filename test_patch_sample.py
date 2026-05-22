import numpy as np

# Load your master tensor dataset via memory mapping
X = np.load("X_final.npy", mmap_mode='r')

# Slice out the very first patch index and save it as a separate file
np.save("test_patch_sample.npy", X[2800])
print("Successfully generated 'test_patch_sample.npy' in your project root!")