
# Image Orientation Classification and Prediction

This project involves building and evaluating Convolutional Neural Network (CNN) models to classify and predict image orientations using various deep learning techniques. The focus is on developing robust models that can classify and predict angular differences between pairs of images, using both custom CNNs and transfer learning approaches with pre-trained models.

## Project Structure

### 1. Data Preparation
- **Cropping and Resizing:** Images are first cropped to remove unnecessary white spaces while retaining the object in the image. Then, the cropped images are resized to a uniform dimension (100x100 for the custom CNN, and 224x224 for transfer learning with MobileNetV2).
- **Pairing Images:** Each image is paired with other images that have a small angular difference (±45 degrees). This results in 31 pairs for each image.
- **Shuffling and Splitting:** The dataset is shuffled and split into training, validation, and test sets. These sets are then further processed to create input data for the CNN models.

### 2. Modeling
- **Custom CNN Model for Classification:**
  - A custom CNN is built with four convolutional layers, followed by dense layers.
  - The model classifies the angular difference between pairs of images into 31 categories (each representing a small angular range).
  - **Training:** The model is trained using categorical cross-entropy as the loss function, and accuracy is tracked across training and validation sets.
  - **Results:** The model achieves high accuracy on both training and validation sets, indicating good generalization.

- **Custom CNN Model for Regression:**
  - A similar CNN architecture is used to predict the angular difference between pairs of images as a continuous value.
  - **Training:** The model is trained using mean squared error (MSE) as the loss function, which is appropriate for regression tasks.
  - **Results:** The MSE on training, validation, and test sets is low, demonstrating the model's ability to accurately predict angular differences.

- **Transfer Learning with MobileNetV2:**
  - **Pre-processing:** Images are converted to grayscale, resized to 224x224, and then paired similarly to the earlier steps.
  - **Fine-tuning MobileNetV2:**
    - The pre-trained MobileNetV2 is used as the base model. The last few layers are unfrozen, and additional convolutional and dense layers are added to fine-tune the model for the specific task.
    - **Training:** The model is trained with a low learning rate to ensure fine-tuning doesn't disrupt the pre-trained weights significantly.
    - **Results:** The model achieves reasonable MSE values, indicating that it effectively transfers knowledge from the pre-trained model to the specific task at hand.

- **Siamese Network with MobileNetV2:**
  - A Siamese network is built using MobileNetV2 as the base model to handle the paired image inputs.
  - **Training:** The network is trained to minimize the MSE between the predicted and actual angular differences between image pairs.
  - **Results:** While the model shows signs of overfitting, the performance is still satisfactory for the task.

### 3. Results and Analysis
- The models demonstrate good performance on the training, validation, and test sets, with low MSE values in regression tasks and high accuracy in classification tasks.
- The project highlights the effectiveness of CNNs for image-based classification and regression tasks, as well as the advantages of transfer learning for improving performance with pre-trained models.

### 4. Future Work
- **Further Fine-tuning:** Additional fine-tuning and hyperparameter optimization could reduce overfitting in the Siamese network.
- **Exploration of Other Architectures:** Exploring other pre-trained models or custom architectures could yield further improvements in performance.
- **Augmentation:** Implementing more data augmentation techniques could improve the model's robustness and generalization capabilities.
