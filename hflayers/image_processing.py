from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

def tensor_to_image(image_vector):
    # Reshape the flattened tensor (4096,) back to (64, 64)
    image_tensor = image_vector.view(64, 64)

    # Convert to NumPy array
    image_array = image_tensor.numpy()

    # Matplotlib expects pixel values in the range [0,1] or [0,255]
    plt.imshow(image_array, cmap='gray')
    plt.axis('off')  # Hide axes
    plt.show()





# Function to load and process an image
def process_image(image_path):
    # Load the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale

    # Define the transformation (resize, convert to tensor, and normalize)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize to 64x64 pixels
        transforms.ToTensor(),        # Convert image to tensor (values between 0 and 1)
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to range [0, 1]
    ])

    # Apply the transformation
    image_tensor = transform(image)

    # Flatten the image (64x64 becomes a vector of size 4096)
    image_vector = image_tensor.view(-1)  # Flatten the image to a vector

    return image_vector

# Example usage
image_path = 'test1.png'
image_vector = process_image(image_path)

tensor_to_image(image_vector)
print(image_vector)  # This should be torch.Size([4096]) for 64x64 images