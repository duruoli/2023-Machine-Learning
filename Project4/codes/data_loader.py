import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

### 2.Image Classification
## 2.7 load in image & presentation

def load_data(file_path, id_bytes, label_bytes, num_images, height, width, depth):
    """
    Function to load image data from a binary file.

    Parameters:
    file_path (str): Path to the binary file.
    id_bytes (int): Number of bytes used for the image ID.
    label_bytes (int): Number of bytes used for the image label.
    num_images (int): Number of images in the binary file.
    height (int): Height of the images.
    width (int): Width of the images.
    depth (int): Depth of the images (number of color channels).

    Returns:
    images (torch.Tensor): Tensor of image data.
    labels (torch.Tensor): Tensor of image labels.
    """

    # Define the record size in bytes
    record_size = id_bytes + label_bytes + height * width * depth

    # Initialize tensors to store the image IDs, labels, and data
    ids = torch.empty((num_images, id_bytes), dtype=torch.uint8)
    labels = torch.empty((num_images, label_bytes), dtype=torch.uint8)
    images = torch.empty((num_images, depth, height, width), dtype=torch.uint8)

    # Open the binary file and read the contents
    with open(file_path, 'rb') as file:
        for i in range(num_images):
            byte_record = file.read(record_size)

            # Convert the byte string to a tensor
            byte_tensor = torch.tensor(list(byte_record), dtype=torch.uint8)

            # Extract the image ID
            ids[i] = byte_tensor[:id_bytes].view(1, -1)

            # Extract the image label
            labels[i] = byte_tensor[id_bytes:id_bytes + label_bytes].view(1, -1)

            # Extract the image data
            array_image = byte_tensor[id_bytes + label_bytes:record_size].view(depth, height, width)
            images[i] = array_image

    # Ensure labels is a 1-D tensor
    labels = labels[:, -1]

    return images, labels


if __name__ == '__main__':
    # Specify parameters (information can be found in the readme file)
    id_bytes = 4
    label_bytes = 4
    num_train_files = 1
    num_train_images = 50000
    width = 64
    height = 64
    depth = 3
    num_classes = 10

    # Load training and test data
    train_images, train_labels = load_data('binary_ver/data_batch_1.bin', id_bytes, label_bytes, num_train_images, height, width, depth)

    # Split training data into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)

    # Print shapes of datasets
    print(train_images.shape)
    print(train_labels.shape)
    print(val_images.shape)
    print(val_labels.shape)

    # Show the second training image
    plt.imshow(train_images[1].permute(1, 2, 0))
    plt.show()
    
    # Get the unique classes from the training labels
    classes = torch.unique(train_labels)

    # Set the number of images to display per class
    num_images_per_class = 1

    # Iterate over the classes
    for class_label in classes:
        # Get the indices of images belonging to the current class
        indices = torch.nonzero(train_labels == class_label, as_tuple=False).squeeze()

        # Select the first few images from the current class
        images_to_display = train_images[indices[:num_images_per_class]]

        # Iterate over the images and display them
        for image in images_to_display:
            # Convert image tensor to NumPy array
            image_np = image.permute(1, 2, 0).numpy()

            # Display the image
            plt.imshow(image_np)
            plt.title(f"Class: {class_label.item()}")
            plt.axis('off')
            plt.show()
            
