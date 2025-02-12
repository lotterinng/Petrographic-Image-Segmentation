import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from petrographic_image_utils import rgb_to_2d_label

# Define the root directory containing the mask dataset
root_directory = 'petrographic_image_dataset/'

# Collect mask images
mask_dataset = []
for path, subdirs, files in os.walk(root_directory):
    dirname = path.split('/')[-1]
    if dirname == 'masks':  # Find all 'masks' directories
        masks = os.listdir(path)  # List of all image names in this subdirectory
        for mask_name in masks:
            if mask_name.endswith(".tiff"):  # Only read .tiff images
                mask = cv2.imread(os.path.join(path, mask_name), 1)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                mask_dataset.append(mask)

mask_dataset = np.array(mask_dataset)

# Convert masks to label format
labels = []
for i in range(mask_dataset.shape[0]):
    label = rgb_to_2d_label(mask_dataset[i])
    labels.append(label)

labels = np.array(labels)

# Compute histogram
histogram, edges = np.histogram(labels, bins=5)
total_pixels = np.sum(histogram)
percentages = np.around(histogram / total_pixels * 100, decimals=1)

print("Unique labels in label dataset are: ", np.unique(labels))
print("Occurrences per class: ", histogram)
print("Total pixels: ", total_pixels)
print("Occurrence percentages: ", percentages)

# Plot histogram
plt.figure(figsize=(8, 6))
plt.bar(range(len(histogram)), histogram, tick_label=[f"Class{i}" for i in range(len(histogram))], color='royalblue')

# Add labels and title
plt.xlabel("Class Labels")
plt.ylabel("Pixel Count")
plt.title("Histogram of Class Occurrences in Mask Dataset")

# Save the image
plt.savefig("histogram_class_occurrences.png")

# Show the plot
plt.show()
