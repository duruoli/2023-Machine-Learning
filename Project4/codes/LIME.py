import numpy as np
import torch
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
from data_loader import load_data
from sklearn.model_selection import train_test_split

# Define a normalization function
def normalize(data):
    data = data.to(dtype=torch.float64)
    img_mean = data.mean(dim=(0, 2, 3))
    img_std = data.std(dim=(0, 2, 3))
    normalize = Normalize(img_mean, img_std)
    preprocessed_data = normalize(data)
    return preprocessed_data
# Placeholder CNN model for illustration. Replace this with your actual model.
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
        x = x.reshape(x.size(0), -1) #6144
        # x = self.relu(self.fc1(x))
        # x = self.logsoftmax(self.fc2(x))
        x = self.logsoftmax(self.fc1(x))
        return x


# Function to predict model output given images and the model
def predict_fn(images, model):
    images = torch.from_numpy(images).permute(0, 3, 1, 2).float().to(device)
    logits = model(images)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


# Function to rescale image values to be within 0-1 range.
# This is necessary because 'mark_boundaries' function expects image pixel values between 0 and 1.
def rescale_image(image):
    image_min = image.min()
    image_max = image.max()
    image = (image - image_min) / (image_max - image_min + 1e-5)
    return image


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
    
    # Initialize the model
    model = VanillaCNN()

    # Load your pre-trained model weights here
    model.load_state_dict(torch.load('saved_models/E19B20_VanillaCNN.pt'))

    # Set the model to evaluation mode.
    model.eval()

    # Define the device for the model
    device = torch.device('cpu')
    model.to(device)

    # Initialize LIME explainer
    explainer = lime_image.LimeImageExplainer()

    # Here, you will need to replace 'YOUR_IMAGE_TENSOR' with your actual image tensor. Something like norm_train_images[0].
    i=1
    example_image = norm_train_images[i]

    # Convert image to numpy and make it suitable for LIME
    test_image = example_image.permute(1, 2, 0).numpy()

    # Generate explanations
    explanation = explainer.explain_instance(test_image, lambda x: predict_fn(x, model), top_labels=5, hide_color=0, num_samples=1000)

    # Get mask for the first prediction
    # positive_only: Only use "positive" features - ones that increase the prediction probability
    # num_features: The number of superpixels to include in the explanation
    # hide_rest: If true, the non-explanation part of the image is greyed out
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=1,  hide_rest=True)

    # Normalize the image for visualization
    normalized_img = rescale_image(temp)

    # Visualize the explanation
    plt.imshow(mark_boundaries(normalized_img, mask))
    plt.show()  # Show the plot
    image = train_images[i]
    image_np = image.permute(1, 2, 0).numpy()

    # Display the image
    plt.imshow(image_np)
    plt.show()