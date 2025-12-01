import numpy as np
import cv2
import os

FILE_PATH = '../Processed_Data_Full_HighRes/original/000.npy'

if not os.path.exists(FILE_PATH):
    import glob
    files = glob.glob('../Processed_Data_Full_HighRes/*/*.npy')
    if files:
        FILE_PATH = files[0]
    else:
        print("Could not find any .npy files!")
        exit()

print(f"Inspecting: {FILE_PATH}")
data = np.load(FILE_PATH)

print(f"Shape: {data.shape}")
print(f"Min Value: {data.min()}")
print(f"Max Value: {data.max()}")

# Get the first frame
frame = data[0]

# Scaling
if data.max() <= 1.0:
    print("Data is 0-1 float. Scaling to 0-255 for visibility.")
    frame = frame * 255.0

# Save
cv2.imwrite('full_view.jpg', cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))
print("Saved 'full_view.jpg'. Please download and look at it!")