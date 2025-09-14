# Comprehensive Report on Vehicle Damage Claim Fraud Detection

## 1. Problem Statement
The problem is to classify vehicle damage claims as either fraudulent or non-fraudulent based on images of the damaged vehicles. This is a binary classification task where the goal is to build a model that can accurately identify potentially fraudulent claims to assist in the review process.

## 2. Objective
The objective is to develop and evaluate a neural network model that can effectively classify vehicle damage images as 'Fraud' or 'Non-Fraud'. The objective includes exploring transfer learning with different pre-trained CNN architectures and creating an ensemble model to potentially improve performance, especially in handling the class imbalance present in the dataset.

## 3. Data Description
The dataset is located at "/content/drive/MyDrive/data". It is split into training and testing sets within the "train" and "test" folders, respectively.
- **Training Data**: Located in "/content/drive/MyDrive/data/train". It contains images categorized into 'Fraud' and 'Non-Fraud' classes. The class distribution in the training data is highly imbalanced:
    - Fraud samples: 200
    - Non-Fraud samples: 5014
    - Total training samples: 5214
- **Testing Data**: Located in "/content/drive/MyDrive/data/test". It contains images for evaluating the trained models. The class distribution in the testing data is:
    - Fraud samples: 93
    - Non-Fraud samples: 1323
    - Total testing samples: 1416

The images are of varying sizes and formats, and are preprocessed to a target size of 256x256 pixels for model input.

## 4. Approach
The approach taken in this notebook involves the following steps:
- **Data Loading and Preprocessing**: Loading the image datasets using `image_dataset_from_directory`, resizing images, and scaling pixel values.
- **Addressing Class Imbalance**: Analyzing the class distribution and calculating class weights to mitigate the impact of the imbalanced dataset during training.
- **Data Augmentation**: Applying data augmentation techniques (random flips, rotations, zooms, contrast adjustments) to the training data to increase the dataset size and improve model generalization.
- **Transfer Learning**: Utilizing pre-trained Convolutional Neural Network (CNN) models (EfficientNetV2-B0, ResNet50, and ConvNeXt-Tiny) as base models for feature extraction. The base models' weights are frozen during initial training.
- **Model Building**: Building classification models on top of the pre-trained base models by adding global average pooling and a dense output layer with a sigmoid activation for binary classification.
- **Individual Model Training**: Training each of the three models (EfficientNetV2-B0, ResNet50, and ConvNeXt-Tiny) on the training data using the calculated class weights.
- **Ensemble Modeling**: Creating an ensemble model by averaging the predictions of the three trained models.
- **Model Evaluation**: Evaluating the performance of the individual models and the ensemble model on the test data using metrics like accuracy, precision, recall, and F1-score, and visualizing the results with a confusion matrix.
- **Feature Map Visualization**: Visualizing the feature maps of intermediate layers in the base models to understand what features each layer is learning.
- **Model Saving**: Saving the trained ensemble model for potential deployment in a web application.
- **Web Application Development (Planned)**: Outlining the steps and providing code for a Flask web application to allow users to upload images and get predictions from the trained ensemble model.

## 5. Pitfalls Encountered
- **Class Imbalance**: The significant imbalance between 'Fraud' and 'Non-Fraud' classes is a major challenge that can lead to models biased towards the majority class. This was addressed by calculating and applying class weights during training.
- **Correctly Accessing Intermediate Layers for Visualization**: Initially, there were difficulties in accessing and visualizing the outputs of intermediate layers within the pre-trained base models when they were part of a `Sequential` model. This was resolved by creating new functional models that explicitly take the input and output the desired intermediate layer activations directly from the base model objects.
- **Passing Data to Feature Extraction Models**: Errors were encountered when passing the preprocessed image to the feature extraction models due to incorrect handling of the input tensor and data augmentation within the feature model definition. This was corrected by ensuring the feature models were defined with a clear input layer and that the preprocessed image (without augmentation for visualization) was correctly passed for prediction.
- **Incorrect Layer Names for Visualization**: Attempting to access intermediate layers using incorrect names resulted in `ValueError`. This was fixed by inspecting the layer names of the base models and using the correct names for visualization.
- **Dimensionality Mismatch in Visualization**: Trying to visualize non-convolutional layer outputs (like pooling or dense layer outputs) as 3D feature maps led to `IndexError`. This was addressed by checking the shape of the output and visualizing convolutional layer outputs as images and 1D outputs as bar plots.

## 6. Architecture and Pipeline
The overall pipeline involves data loading, preprocessing, data augmentation, building and training individual transfer learning models, ensembling these models, and evaluating the final ensemble.

The architecture for each individual model is a transfer learning approach:
- **Input Layer**: Takes images of size (256, 256, 3).
- **Data Augmentation Layer**: Applies random transformations to the training images (skipped during inference and visualization of learned features).
- **Base Model (EfficientNetV2-B0, ResNet50, or ConvNeXt-Tiny)**: A pre-trained CNN model with weights frozen during initial training. This acts as a powerful feature extractor.
- **Global Average Pooling Layer**: Reduces the spatial dimensions of the feature maps from the base model to a fixed-size vector.
- **Dense Output Layer**: A single neuron layer with a sigmoid activation function to output a probability between 0 and 1, representing the likelihood of the image belonging to the 'Fraud' class.

The ensemble model combines the predictions of the three individual models by averaging their sigmoid outputs.

## 7. Models Used and How They Work
Three different pre-trained CNN models were used for transfer learning:

- **EfficientNetV2-B0**:
    - **How it works**: EfficientNetV2 is a family of models that optimize for both accuracy and training speed. It uses fused-MBConv blocks, which are variations of inverted bottleneck convolution blocks that replace depthwise and pointwise convolutions with a single standard convolution in the initial layers for better efficiency. It also employs progressive learning during training.
    - **What each filter layer is learning**: Early layers learn basic features like edges, corners, and textures. Deeper layers learn more complex and abstract representations, capturing higher-level semantic information relevant to the image content. The 'top_activation' layer before pooling provides a rich set of high-level features.

- **ResNet50**:
    - **How it works**: ResNet (Residual Network) uses residual connections (skip connections) that allow the gradient to flow more easily through the network during training. This helps in training very deep networks without suffering from the vanishing gradient problem. ResNet50 uses bottleneck residual blocks to reduce computational cost.
    - **What each filter layer is learning**: Similar to other CNNs, early layers detect simple features. Intermediate and deeper layers, aided by residual connections, learn more complex and robust features that are combinations of the earlier features. The 'conv5_block3_add' layer, being deep in the network, captures highly abstract features.

- **ConvNeXt-Tiny**:
    - **How it works**: ConvNeXt is a modern CNN architecture that incorporates design principles from Vision Transformers while maintaining the efficiency of CNNs. It uses large kernel convolutions, inverted bottleneck structures, and fewer activation and normalization layers compared to traditional CNNs.
    - **What each filter layer is learning**: ConvNeXt layers learn hierarchical features. Early layers capture local patterns. As the network deepens, the larger kernel sizes in depthwise convolutions allow layers to learn features over a wider spatial context. The 'norm' layer before pooling aggregates these learned features.

## 8. Activation Function
The primary activation function used in the final output layer for binary classification is the **sigmoid function**.
- **Sigmoid Function**:
    - **Mathematical Formula**: σ(x) = 1 / (1 + exp(-x))
    - **Purpose**: The sigmoid function squashes the output of the dense layer to a value between 0 and 1, which can be interpreted as the probability of the input image belonging to the positive class (Fraud in this case).

Other activation functions like **ReLU (Rectified Linear Unit)** and **GELU (Gaussian Error Linear Unit)** are used within the intermediate layers of the base models.
- **ReLU Function**: max(0, x) - introduces non-linearity.
- **GELU Function**: Approximately x * Φ(x), where Φ(x) is the cumulative distribution function of the standard normal distribution - a smoother approximation to ReLU, often used in newer architectures like ConvNeXt.

## 9. Normalization
**Batch Normalization** is used extensively within the layers of EfficientNetV2-B0 and ResNet50.
- **Batch Normalization**: Normalizes the activations of a layer across the mini-batch. This helps in stabilizing the training process, allowing for higher learning rates and acting as a regularizer.

**Layer Normalization** is used in ConvNeXt-Tiny.
- **Layer Normalization**: Normalizes the activations across the features within a single sample. This is often preferred in architectures like Transformers and newer CNNs like ConvNeXt.

## 10. How the Ensemble Model Works
The ensemble model combines the predictions of the three individually trained models (EfficientNetV2-B0, ResNet50, and ConvNeXt-Tiny). In this case, a simple **averaging ensemble** is used.
- **Process**: For a given input image, each of the three trained models makes a prediction (a probability between 0 and 1). The ensemble model then calculates the average of these three probabilities. This averaged probability is the final prediction of the ensemble model.
- **Benefit**: Ensembling can often improve performance compared to individual models by reducing variance and potentially capturing different aspects of the data that individual models might miss. If different models make uncorrelated errors, averaging their predictions can lead to a more robust and accurate final prediction.

## 11. Evaluation Scores
The models were evaluated on the test dataset. The key evaluation scores obtained are:

- **EfficientNetV2-B0**:
    - Loss on test data: Approximately 0.2269
    - Accuracy on test data: Approximately 0.9258

- **ResNet50**:
    - Loss on test data: Approximately 0.3135
    - Accuracy on test data: Approximately 0.8771

- **ConvNeXt-Tiny**:
    - Loss on test data: Approximately 0.2543
    - Accuracy on test data: Approximately 0.9343

- **Ensemble Model**:
    - Loss on test data: Approximately 0.2355
    - Accuracy on test data: Approximately 0.9393

The ensemble model achieved the highest accuracy on the test data compared to the individual models.

## 12. Deeper Analysis of Evaluation Metrics and Confusion Matrix
The classification report and confusion matrix provide a more detailed view of the ensemble model's performance beyond just accuracy.

- **Classification Report**:
    - Provides precision, recall, and F1-score for each class ('Fraud' and 'Non-Fraud').
    - **Precision**: The ability of the model to not label a negative sample as positive (correct positive predictions out of all positive predictions).
    - **Recall (Sensitivity)**: The ability of the model to find all the positive samples (correct positive predictions out of all actual positive samples).
    - **F1-score**: The harmonic mean of precision and recall, providing a balance between the two metrics.
    - **Support**: The number of actual samples in each class in the test set.

    *Based on the provided output:*
    - **Fraud Class (Minority)**:
        - Precision: 0.53 (Out of all samples predicted as Fraud, 53% were actually Fraud).
        - Recall: 0.76 (Out of all actual Fraud samples, 76% were correctly identified).
        - F1-score: 0.62 (A balance between precision and recall for the Fraud class).
        - Support: 93
    - **Non-Fraud Class (Majority)**:
        - Precision: 0.98 (Out of all samples predicted as Non-Fraud, 98% were actually Non-Fraud).
        - Recall: 0.95 (Out of all actual Non-Fraud samples, 95% were correctly identified).
        - F1-score: 0.97
        - Support: 1323
    - **Overall Accuracy**: 0.94

- **Confusion Matrix**:
    - A table that visualizes the performance of a classification model. Each row represents the instances in an actual class, while each column represents the instances in a predicted class.
    - **True Positives (TP)**: Correctly predicted Fraud samples.
    - **True Negatives (TN)**: Correctly predicted Non-Fraud samples.
    - **False Positives (FP)**: Incorrectly predicted Fraud samples (Type I error).
    - **False Negatives (FN)**: Incorrectly predicted Non-Fraud samples (Type II error).

    *Based on the provided output:*
    - True Positives (Fraud correctly predicted): 71
    - False Negatives (Fraud incorrectly predicted as Non-Fraud): 22
    - False Positives (Non-Fraud incorrectly predicted as Fraud): 64
    - True Negatives (Non-Fraud correctly predicted): 1259

    - The confusion matrix highlights the trade-off between correctly identifying fraudulent claims (high recall for Fraud) and minimizing false alarms (high precision for Fraud).
    - While the overall accuracy is high, the lower precision for the 'Fraud' class indicates that a notable number of non-fraudulent claims are being flagged as potentially fraudulent (False Positives). The relatively higher recall for 'Fraud' means the model is reasonably good at catching actual fraudulent cases.
    - In a fraud detection scenario, the cost of a False Negative (missing a fraudulent claim) is often higher than the cost of a False Positive (reviewing a legitimate claim). The current model has a good recall for the minority class, which is a positive sign for identifying potential fraud, but the number of false positives suggests that further optimization or a different threshold might be needed depending on the cost associated with each type of error.

## 13. Feature Map Visualizations
The visualizations of feature maps from intermediate layers of the base models (EfficientNetV2-B0, ResNet50, ConvNeXt-Tiny) show how the networks process the input image and learn hierarchical representations.

- **Early Layers**: Feature maps from early layers (e.g., 'block2b_add' in EfficientNetV2-B0, 'conv2_block2_add' in ResNet50, 'convnext_tiny_stage_0_block_0_pointwise_conv_1' in ConvNeXt-Tiny) tend to highlight basic visual features like edges, lines, and simple textures. The activations in these layers often correspond to the presence and orientation of these fundamental elements in the image.
- **Mid-level Layers**: Feature maps from mid-level layers (e.g., 'block4b_add' in EfficientNetV2-B0, 'conv3_block3_add' in ResNet50, 'convnext_tiny_stage_1_block_1_pointwise_conv_1' in ConvNeXt-Tiny) show more complex patterns. These layers learn to combine the basic features from earlier layers to detect more elaborate structures, such as corners, curves, or parts of objects.
- **Deeper Layers**: Feature maps from deeper layers (e.g., 'block6b_add' and 'top_activation' in EfficientNetV2-B0, 'conv4_block5_add' and 'conv5_block3_add' in ResNet50, 'convnext_tiny_stage_2_block_2_pointwise_conv_1' and 'norm' in ConvNeXt-Tiny) represent highly abstract and semantic features. These layers have learned to identify more complex patterns and objects relevant to the task. For a fraud detection task, these layers might be sensitive to patterns of damage, vehicle parts, or contextual cues that differentiate fraudulent from non-fraudulent claims.
- **Pooling Features**: The pooling features (obtained after global average pooling) are condensed representations of the high-level features. The bar plots of these features show the activation strength of different abstract features learned by the network. Analyzing these plots can provide insights into which high-level features are most prominent in a given image and how they contribute to the final classification decision. Comparing pooling features for 'Fraud' and 'Non-Fraud' images could reveal discriminative patterns.

Visualizing these feature maps helps in understanding the "black box" nature of neural networks and provides some intuition about what the models are focusing on to make predictions. It confirms that the models are learning meaningful visual hierarchies relevant to the image content.

## 14. Code-Level Details for Web Application
The provided `app.py` code outlines the structure for a Flask web application.

- **Loading the Model**: The code loads the saved ensemble model (`ensemble_model.keras`) using `tf.keras.models.load_model()`. Error handling is included in case the model file is not found.
- **Handling Image Uploads**: The `/predict` route handles POST requests containing an image file. It checks if a file is present and if its extension is allowed (png, jpg, jpeg).
- **Preprocessing**: The uploaded image is saved to a temporary file, loaded using `tf.keras.utils.load_img()`, resized to the target size (256x256), converted to a NumPy array, and scaled to the range [0, 1].
- **Prediction**: The preprocessed image (with an added batch dimension) is passed to the loaded `ensemble_model.predict()` method to obtain the prediction probability.
- **Displaying Results**: The prediction probability is converted into a predicted class ('Fraud' or 'Non-Fraud') based on a threshold of 0.5. The predicted class and probability are then passed to a `result.html` template for display to the user. Temporary files are removed after processing.

To run this web application, you would also need an `index.html` file for the image upload form and a `result.html` file to display the prediction. The Flask development server is used for local testing (`app.run(debug=True)`), but a production-ready WSGI server would be required for deployment.

## 15. Crucial Details and Further Considerations
- **Data Augmentation during Inference**: It's important to note that data augmentation is typically applied only during the training phase to expose the model to variations of the training data. During inference (prediction) and for visualizing learned features on a specific image, data augmentation is not applied.
- **Transfer Learning Strategy**: In this notebook, the base models were initially trained with their weights frozen. For potentially better performance, especially with a larger dataset or when the new task is significantly different from the pre-training task, fine-tuning (unfreezing some or all layers of the base model and training them with a small learning rate) could be explored.
- **Hyperparameter Tuning**: The number of epochs, batch size, optimizer learning rate, and data augmentation parameters were set to default or reasonable values. Fine-tuning these hyperparameters could further improve model performance.
- **Ensemble Methods**: While simple averaging was used, other ensembling techniques like weighted averaging (where models are weighted based on their individual performance) or stacking (training a meta-model on the predictions of the base models) could be investigated.
- **Thresholding**: The prediction threshold of 0.5 for binary classification can be adjusted based on the specific needs of the application and the desired trade-off between precision and recall. For instance, if minimizing false negatives is critical, a lower threshold might be chosen, even if it increases false positives.
- **Interpretability Beyond Feature Maps**: While feature map visualization provides insights, other interpretability techniques like Grad-CAM or SHAP values could be used to understand which parts of the input image are most important for the model's prediction.
- **Deployment Considerations**: For a production web application, security, scalability, and efficiency are crucial. Using a robust WSGI server, optimizing model loading and inference time, and potentially deploying on a cloud platform would be necessary.
- **Continuous Monitoring and Retraining**: In a real-world scenario, the model's performance should be continuously monitored, and the model should be retrained periodically with new data to maintain its effectiveness as the data distribution may change over time.