import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torchvision.transforms import Normalize
from data_loader import load_data
import pandas as pd

# Define a normalization function
def normalize(data):
    data = data.to(dtype=torch.float32)
    img_mean = data.mean(dim=(0, 2, 3))
    img_std = data.std(dim=(0, 2, 3))
    normalize = Normalize(img_mean, img_std)
    preprocessed_data = normalize(data)
    return preprocessed_data

class VanillaGrad:
    """ Class for computing gradients of the output w.r.t an input image for a pretrained model """

    def __init__(self, pretrained_model, cuda=False):
        self.pretrained_model = pretrained_model
        self.cuda = cuda

    def __call__(self, x, index=None):
        x.requires_grad_(True)
        output = self.pretrained_model(x)

        # If no index is provided, select the class with the highest probability
        if index is None:
            index = output.argmax(dim=1).item()

        one_hot = torch.zeros_like(output)
        one_hot[:, index] = 1 #change to a certain answer instead of multiple non-zero probability
        one_hot = one_hot.to(x.device) if self.cuda else one_hot

        # Zero gradients
        self.pretrained_model.zero_grad()
        output.backward(gradient=one_hot)# emphasize the gradients which "lead to" the target class
        grad = x.grad.data

        return grad


class SmoothGrad(VanillaGrad):
    """ Class for computing SmoothGrad, which averages gradients of the output w.r.t input image over
        multiple noisy versions of the input """

    def __init__(self, pretrained_model, cuda=False, stdev_spread=0.15, n_samples=25, magnitude=True):
        super(SmoothGrad, self).__init__(pretrained_model, cuda)
        self.stdev_spread = stdev_spread
        self.n_samples = n_samples
        self.magnitude = magnitude

    def __call__(self, x, index=None):
        stdev = self.stdev_spread * (x.max() - x.min())
        total_gradients = torch.zeros_like(x)

        for _ in range(self.n_samples):
            noise = torch.normal(0, stdev, size=x.shape).to(x.device) if self.cuda else torch.normal(0, stdev, size=x.shape)
            x_plus_noise = x + noise

            grad = super(SmoothGrad, self).__call__(x_plus_noise, index=index)

            if self.magnitude:
                total_gradients += (grad * grad)
            else:
                total_gradients += grad

        avg_gradients = total_gradients / self.n_samples
        return avg_gradients


# Note: This is an overly simplified model and is meant for demonstration purposes only.
# It is used here as a placeholder. You should use what you've built for the previous questions.

class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2) #decrease the size of imgae
        # And a fully connected layer like this:
        self.fc1 = nn.Linear(in_features=24*16*16, out_features=num_classes) #two maxpool(size=2): 64*64=>16*16
        #self.fc2 = nn.Linear(128, out_features=num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        # Continue to define your network here...

    def forward(self, x):
        # Define the forward pass of your network here.
        x = self.pool(self.relu(self.conv1(x))) 
        x = self.pool(self.relu(self.conv2(x))) 
        x = x.view(x.size(0), -1) #6144
        # x = self.relu(self.fc1(x))
        # x = self.logsoftmax(self.fc2(x))
        x = self.logsoftmax(self.fc1(x))
        return x




if __name__ == '__main__':
    # Specify parameters (information can be found in the readme file)
    id_bytes = 4
    label_bytes = 4
    num_train_files = 1
    num_train_images = 5000
    width = 64
    height = 64
    depth = 3
    num_classes = 10    
    # Load training and test data
    train_images, train_labels = load_data('binary_ver/data_batch_1.bin', id_bytes, label_bytes, num_train_images, height, width, depth)

    # Split training data into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)

    norm_train_images = normalize(train_images)
    norm_val_images = normalize(val_images)
    
    
    # Initialize your model by calling the class that defines your model architecture.
    # Here, 'ExampleCNN' is a placeholder for your model choice.
    model = VanillaCNN()

    # The weights from your pretrained model should be saved in a .pt file.
    model_weights = 'saved_models/E19B20_VanillaCNN.pt'

    # Uncomment the following line to load the weights into the model.
    # 'torch.load' will load the weights, and 'model.load_state_dict' will apply these weights to your model.
    # Make sure that the architecture of 'model' matches the architecture of the model that the weights came from.
    model.load_state_dict(torch.load(model_weights))

    # Set the model to evaluation mode.
    # This step is necessary because it tells your model that it will be used for inference, not training.
    # In evaluation mode, certain layers like dropout are disabled.
    model.eval()

    # Initialize SmoothGrad. It will average the gradients over 25 noisy versions of the input. Each noisy version is
    # obtained by adding Gaussian noise to the input with a standard deviation of 15% of the input's range.
    # You can change these numbers to vary noise levels and number of images for averaging.
    smooth_grad = SmoothGrad(pretrained_model=model, cuda=False, stdev_spread=0.1, n_samples=100, magnitude=True)

    # Here, you will need to replace 'YOUR_IMAGE_TENSOR' with your actual image tensor. Something like norm_train_images[0].
    i = 0
    example_image = norm_train_images[i]

    # Compute the SmoothGrad saliency map
    # The image tensor is unsqueezed to add an extra dimension because the model expects a batch of images.
    # The dtype is set to float32, as the model expects input data in this format.
    smooth_saliency = smooth_grad(example_image.to(dtype=torch.float32).unsqueeze(0))

    # Convert the saliency map to absolute values, because we are interested in the magnitude of the gradients,
    # regardless of their direction.
    abs_saliency = np.abs(smooth_saliency.numpy())

    # Sum the absolute gradients across all color channels to get a single saliency map.
    # 'squeeze' is used to remove the extra dimension that was added earlier.
    saliency_map = np.sum(abs_saliency, axis=1).squeeze()

    # Display the final saliency map. The brighter a pixel in the saliency map, the more important it is for the model's decision.
    plt.imshow(saliency_map, cmap='gray')
    plt.show()

    image = train_images[i]
    image_np = image.permute(1, 2, 0).numpy()

    # Display the image
    plt.imshow(image_np)
    plt.show()
