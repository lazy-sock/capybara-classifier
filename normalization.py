import numpy as np
import cv2
import os

def compute_mean_std(dataset_path):
    means = []
    stds = []
    image_count = 0
    for bird_class in os.listdir(dataset_path):
        for filename in os.listdir(dataset_path + '/' + bird_class):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(dataset_path + '/' + bird_class, filename)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image / 255.0  # Scale pixel values to [0, 1]

                means.append(np.mean(image, axis=(0, 1)))
                stds.append(np.std(image, axis=(0, 1)))
                image_count += 1

    # Compute average mean and std across all images
    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)

    return mean, std

dataset_path = 'bird_dataset_v3'
mean, std = compute_mean_std(dataset_path)
print(f'Mean: {mean}, Std: {std}')
