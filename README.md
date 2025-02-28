# Home_Assignment-2_CNN
Overview
This repository contains multiple Python scripts that demonstrate various deep learning and image processing techniques using TensorFlow, OpenCV, and NumPy. The scripts cover:
2D Convolution using TensorFlow – Applying a 3×3 kernel to a 5×5 matrix with different strides and padding techniques.
Sobel Edge Detection using OpenCV – Detecting edges in an image using Sobel filters.
Max Pooling and Average Pooling using TensorFlow – Performing downsampling on a 4×4 matrix using max pooling and average pooling.
AlexNet Implementation using TensorFlow – Constructing and summarizing the AlexNet deep learning architecture.
ResNet-like Model with Residual Blocks – Implementing a simplified ResNet architecture with skip connections.

1. 2D Convolution using TensorFlow

Description

This script applies a 3×3 kernel (Laplacian operator) to a 5×5 matrix to demonstrate the convolution operation with different stride and padding values.

Steps

Define a 5×5 input matrix.
Define a 3×3 kernel for edge detection.
Convert matrices to TensorFlow tensors.
Apply tf.nn.conv2d() with different strides and padding options.
Print the output matrices.

Expected Output

The script prints the convolved matrices for each combination of stride (1,2) and padding (VALID, SAME).

2. Sobel Edge Detection using OpenCV

Description

This script applies Sobel filters to detect edges in an image.

Steps

Load a grayscale image using OpenCV.
Apply cv2.Sobel() to compute gradients in both the x and y directions.
Convert results to absolute values for visualization.
Display the original image, Sobel-X, and Sobel-Y outputs using Matplotlib.

Expected Output

The script displays:
The original grayscale image.
The edges detected in the x-direction.
The edges detected in the y-direction.

3. Max Pooling and Average Pooling using TensorFlow

Description

This script applies max pooling and average pooling operations on a random 4×4 matrix.

Steps

Generate a random 4×4 matrix.
Define MaxPooling2D and AveragePooling2D layers with a 2×2 kernel and stride of 2.
Apply both pooling operations.
Print the original and pooled matrices.

Expected Output

The script prints:
Original 4×4 matrix.
Max Pooled matrix (2×2).
Average Pooled matrix (2×2).

4. AlexNet Implementation using TensorFlow

Description

This script defines the AlexNet architecture, a deep learning model for image classification.

Model Structure
5 Convolutional layers with ReLU activation.
3 MaxPooling layers for downsampling.
Flattening layer to convert feature maps into a vector.
Fully connected (Dense) layers with dropout.
Softmax output layer for classification.

Steps

Construct the AlexNet model using TensorFlow's Sequential().
Print the model summary using model.summary().

Expected Output

A table showing each layer’s name, output shape, and number of parameters.

ResNet-like Model with Residual Blocks

Description

This script defines a simplified ResNet-style model with residual blocks.

Model Structure

A 7×7 Conv2D layer with stride 2 for initial feature extraction.
Two residual blocks, where the input is added to the output to help gradient flow.
Flattening and Fully Connected layers for classification.
Softmax output layer for multi-class classification.

Steps

Define a residual block function that adds a skip connection.
Construct a simple ResNet model using Functional API.
Print the model summary.

Expected Output

A table showing each layer’s name, output shape, and number of parameters.
