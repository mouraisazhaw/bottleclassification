Project Name: Bottle Classification

Created by: Isabella Moura, WIN21HSTZDSB, mouraisa@students.zhaw.ch

1. Project Goal/Motivation

Motivation:
The motivation behind this project is to develop a model for bottle classification. This is an important application in the industry, especially in the field of automation, quality control and recycling. Classifying bottles can help identify defective products and make the production process more efficient. It could also represent a solution to enable automated recycling of bottles.

Problem Statement:
The main problem is to find an accurate and reliable method for identifying and classifying bottles. Traditional methods are often time-consuming and error-prone. Therefore, the use of transfer learning and modern machine learning techniques is proposed to improve accuracy and efficiency.

Relevance:
This project is relevant because it demonstrates how advanced machine learning and image processing techniques can be applied in real-world industrial applications. Improving classification accuracy can have a direct impact on production quality and costs.

2. Data Collection or Generation

The data for this project was collected from Kaggle. The dataset contains 25k images of bottles in various categories. The link to the dataset used is: https://www.kaggle.com/datasets/vencerlanz09/bottle-synthetic-images-dataset/data.

The data collection process involved several steps:

	1.	Downloading the image data.
	2.	Preprocessing the images, including resizing, augmenting and normalization.
	3.	Splitting the data into training, validation, and test sets.

3. Modeling

Transfer learning was used for modeling to optimize the performance of the model. The specific model used in this project is a pre-trained model that was further fine-tuned and customized. This code sets up a transfer learning pipeline using the Xception model pre-trained on ImageNet, adding custom layers on top to adapt it for the classification task. The base model’s weights are frozen to leverage learned features, while the new layers are trained to fit the new dataset.

Modeling Steps:

	1.	Selection of a suitable pre-trained model (Xception model, pre-trained on ImageNet).
	2.	Customizing the final layers of the model.
	3.	Fine-tuning the model using training data and data augmentation techniques.
	4.	Training the model with the optimized dataset.

4. Interpretation and Validation

Validation:

The validation of the model was carried out by:

	1.	Using a separate validation dataset during training.
	2.	Performing cross-validation to check the robustness of the model.
	3.	Evaluating the model on an independent test dataset to determine its final performance.
    4.  Analysing Classification Report
    5.  Analysing Confusion Matrix


Interpretation:

### Visualizing Feature Maps: Visualization of Feature maps from different layers to understand what features the model is learning.


### Fine-tuned Model

- Fine-tuning the model after the initial training improves both training and validation accuracy, demonstrating that the model benefits from additional training of the base layers.
- The validation loss decreases notably after fine-tuning, suggesting that the model generalizes better to the validation set once the base layers are unfrozen and trained with a lower learning rate.
- The increase in both training and validation accuracy and the decrease in loss indicate that fine-tuning enhances the model’s performance on the specific task.


### Model Comparison
- The initial model leverages the pre-trained Xception network, focusing on training only the newly added layers.
- Fine-tuning involves retraining the entire network with a lower learning rate to refine the weights further, leading to potentially better performance by adjusting all layers, including the pre-trained ones.
- Structure: Both models have the same structure, with the difference being the trainability of the base Xception layers.
- Training: The initial model trains only the top layers, while the fine-tuned model trains the entire network.
- Compilation: Both models use the same loss function and metric, but the fine-tuned model uses a lower learning rate to avoid large updates that could disrupt the pre-trained weights.


### Summary

- The base model with frozen layers provides a solid starting point, showing reasonable performance and stable validation metrics.
- Fine-tuning the model further improves its accuracy and reduces the loss, demonstrating the effectiveness of additional training on the pre-trained layers with a lower learning rate.
- The validation metrics after fine-tuning suggest that the model’s ability to generalize has improved, reducing overfitting and enhancing overall performance.

- Overall Accuracy: The model achieves an accuracy of 91%.

Class Performance:
- The model performs best on Plastic Bottles with the highest precision (0.98) and recall (0.99).
- Soda Bottles have the lowest recall (0.85), indicating more confusion with other classes.

- Misclassifications: Most misclassifications occur between similar types of bottles, especially Soda Bottles and Beer Bottles.
- F1-Scores: Indicate a balanced performance across precision and recall for each class, with the lowest being 0.87 for Beer Bottles and Soda Bottles.

The model demonstrates strong performance overall, with particularly high precision and recall for Plastic Bottles and Water Bottles. The confusion matrix and classification report suggest areas where the model could improve, particularly in distinguishing between Soda Bottles and other classes.


### APP.PY WebApp using Streamlit: 

Users can upload an image of a bottle, which the app preprocesses and then classifies using the fine-tuned model. The model predicts the type of bottle. The app displays the uploaded image, the predicted bottle type, and the prediction probabilities, providing an easy-to-use interface for bottle classification.
